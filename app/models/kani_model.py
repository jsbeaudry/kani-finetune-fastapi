"""
Kani TTS - Inference Engine
============================

Main inference class that orchestrates the complete text-to-speech pipeline:

    Text prompt  -->  Tokenize  -->  Causal LM generate  -->  Audio tokens  -->  NeMo decode  -->  WAV

Key Optimizations (from the fast-inference notebook)
-----------------------------------------------------

1. **Frame-level position encoding**: All 4 codec tokens within an audio
   frame share the same RoPE position ID. This reduces the effective
   positional distance by 4x, improving long-form generation coherence
   and KV-cache efficiency.

2. **Flash Attention 2**: Loaded at model init time for faster attention
   and lower memory usage.

3. **torch.compile** (optional): Kernel fusion for additional throughput
   on Ampere+ GPUs.

Token Flow
----------

::

    Input:
    [START_OF_HUMAN] + tokenized_text + [END_OF_TEXT, END_OF_HUMAN]

    Model generates:
    [START_OF_AI] + text_echo + [END_OF_TEXT]
    + [START_OF_SPEECH] + audio_tokens + [END_OF_SPEECH]
    + [END_OF_AI]

Speaker Conditioning
--------------------

When a ``speaker_id`` is provided, it is prepended to the text in
lowercase: ``"<speaker_id>: <text>"``. The model must have been
fine-tuned with this speaker ID in the training data.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import Settings
from app.models.audio_player import NemoAudioPlayer


class KaniModel:
    """
    Optimized Kani TTS inference engine with frame-level position encoding.

    Manages the causal language model lifecycle and provides a simple
    ``run_model(text, speaker_id)`` interface that returns a numpy waveform.

    Args:
        config: Application settings (model name, device, generation params).
        player: NemoAudioPlayer instance for token-to-waveform conversion.

    Attributes:
        model: The loaded AutoModelForCausalLM (bfloat16).
        tokenizer: Matching HuggingFace tokenizer.
        device: ``"cuda"`` or ``"cpu"``.

    Example::

        config = Settings(model_name="jsbeaudry/haitian-kani-ht-v3")
        player = NemoAudioPlayer(config)
        kani  = KaniModel(config, player)

        audio, text = kani.run_model(
            "Bonjour, comment allez-vous?",
            speaker_id="alice",
            temperature=0.6,
        )
        # audio: float32 numpy array at 22050 Hz
    """

    def __init__(self, config: Settings, player: NemoAudioPlayer) -> None:
        self.conf = config
        self.player = player
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=self.conf.device_map,
        )
        if self.conf.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.conf.model_name, **load_kwargs
        )

        if self.conf.use_torch_compile:
            self.model = torch.compile(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.model_name)

    def get_input_ids(
        self, text_prompt: str, speaker_id: str = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a text prompt and wrap it in the conversation control tokens.

        Produces the input sequence::

            [START_OF_HUMAN] + BPE_tokens(text) + [END_OF_TEXT, END_OF_HUMAN]

        If ``speaker_id`` is set, the text is first rewritten as
        ``"<speaker_id>: <text>"`` before tokenization.

        Args:
            text_prompt: Raw text to synthesize.
            speaker_id: Optional speaker identity label.

        Returns:
            Tuple of (input_ids, attention_mask, position_ids), each as
            [1, seq_len] tensors on CPU (moved to device in model_request).
        """
        START_OF_HUMAN = self.player.start_of_human
        END_OF_TEXT = self.player.end_of_text
        END_OF_HUMAN = self.player.end_of_human

        if speaker_id is not None:
            text_prompt = f"{speaker_id.lower()}: {text_prompt}"

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        position_ids = torch.arange(
            modified_input_ids.shape[1], dtype=torch.int64
        ).unsqueeze(0)

        return modified_input_ids, attention_mask, position_ids

    def build_frame_position_ids(
        self, generated_ids: torch.Tensor, prefix_len: int
    ) -> torch.Tensor:
        """
        Build frame-level position IDs for a fully generated sequence.

        Position assignment strategy:
        - **Text / control tokens**: sequential positions (0, 1, 2, ...).
        - **Audio tokens**: every group of 4 tokens (one audio frame)
          shares the same position ID. This compresses the RoPE distance
          by 4x within audio regions.
        - **Suffix tokens** (after END_OF_SPEECH): continue sequentially
          from the last audio position.

        Args:
            generated_ids: Full generated tensor [1, total_seq_len].
            prefix_len: Length of the input prefix (text tokens) that
                already have sequential positions.

        Returns:
            Position ID tensor [1, total_seq_len] on the same device.
        """
        seq_len = generated_ids.shape[1]
        position_ids = torch.zeros(
            1, seq_len, dtype=torch.long, device=generated_ids.device
        )

        # Text prefix: sequential
        position_ids[0, :prefix_len] = torch.arange(prefix_len)

        # Find the audio region boundaries
        flat = generated_ids.flatten()
        start_speech = self.player.start_of_speech
        start_indices = (flat == start_speech).nonzero(as_tuple=True)[0]

        if len(start_indices) > 0:
            audio_start = start_indices[0].item() + 1  # after START_OF_SPEECH
            audio_end = seq_len

            end_speech = self.player.end_of_speech
            end_indices = (flat == end_speech).nonzero(as_tuple=True)[0]
            if len(end_indices) > 0:
                audio_end = end_indices[0].item()

            # Control tokens between prefix and audio (START_OF_AI, START_OF_SPEECH)
            for i in range(prefix_len, audio_start):
                position_ids[0, i] = i

            # Audio: groups of 4 share one position
            num_audio_tokens = audio_end - audio_start
            audio_base_pos = audio_start
            for t in range(num_audio_tokens):
                frame_idx = t // 4
                position_ids[0, audio_start + t] = audio_base_pos + frame_idx

            # Suffix: continue from last audio position
            if audio_end < seq_len:
                last_audio_pos = audio_base_pos + (num_audio_tokens - 1) // 4
                for i in range(audio_end, seq_len):
                    position_ids[0, i] = last_audio_pos + 1 + (i - audio_end)
        else:
            # Fallback: no audio found, use sequential positions
            position_ids[0, :] = torch.arange(seq_len)

        return position_ids

    def model_request(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        repetition_penalty: float = None,
    ) -> torch.Tensor:
        """
        Run the causal LM's generate() with sampling parameters.

        Args:
            input_ids: Tokenized input [1, seq_len].
            attention_mask: Binary mask [1, seq_len].
            position_ids: RoPE positions [1, seq_len].
            temperature: Sampling temperature (overrides config default).
            top_p: Nucleus sampling threshold (overrides config default).
            max_new_tokens: Generation length limit (overrides config default).
            repetition_penalty: Token repetition penalty (overrides config default).

        Returns:
            Generated token IDs tensor [1, total_seq_len] on self.device.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        position_ids = position_ids.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens or self.conf.max_new_tokens,
                do_sample=True,
                temperature=temperature or self.conf.temperature,
                top_p=top_p or self.conf.top_p,
                repetition_penalty=repetition_penalty or self.conf.repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.player.end_of_speech,
            )
        return generated_ids.to(self.device)

    def run_model(
        self,
        text: str,
        speaker_id: str = None,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        repetition_penalty: float = None,
    ) -> tuple:
        """
        End-to-end TTS: text in, audio out.

        This is the main public method called by the /tts endpoint.

        Args:
            text: The text to synthesize.
            speaker_id: Optional speaker conditioning label.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling override.
            max_new_tokens: Max generation length override.
            repetition_penalty: Repetition penalty override.

        Returns:
            Tuple of (audio, text) where:
            - audio: float32 numpy array of PCM samples at 22 050 Hz.
            - text: The original input text.

        Raises:
            ValueError: If the model fails to produce valid speech tokens.
        """
        input_ids, attention_mask, position_ids = self.get_input_ids(text, speaker_id)
        model_output = self.model_request(
            input_ids,
            attention_mask,
            position_ids,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        audio, _ = self.player.get_waveform(model_output)
        return audio, text

    def reload_model(self, model_path: str) -> None:
        """
        Hot-swap the loaded model with a different checkpoint.

        Frees the current model's GPU memory before loading the new one.
        Both the model weights and tokenizer are replaced.

        Args:
            model_path: Local path or HuggingFace repo ID for the new model.

        Side effects:
            - Updates ``self.model``, ``self.tokenizer``.
            - Updates ``self.conf.model_name`` to reflect the new model.
            - Calls ``torch.cuda.empty_cache()`` to reclaim VRAM.
        """
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=self.conf.device_map,
        )
        if self.conf.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        del self.model
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if self.conf.use_torch_compile:
            self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.conf.model_name = model_path
