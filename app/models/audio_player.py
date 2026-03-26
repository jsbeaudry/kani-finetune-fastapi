"""
Kani TTS - NeMo Audio Player
==============================

Handles the conversion between model-generated token sequences and playable
audio waveforms using NVIDIA's NeMo Nano Codec (22 kHz, 0.6 kbps, 12.5 fps).

Token Layout
------------

The Kani model uses a unified vocabulary that spans three regions:

    +--------------------------+-------------------------------------------+
    | Token range              | Purpose                                   |
    +==========================+===========================================+
    | 0 .. tokeniser_length-1  | Standard text tokens (BPE)                |
    +--------------------------+-------------------------------------------+
    | tokeniser_length + 1..7  | Special control tokens (see table below)  |
    +--------------------------+-------------------------------------------+
    | tokeniser_length + 10 .. | Audio codec tokens (4 codebooks x 4032)   |
    +--------------------------+-------------------------------------------+

Special control tokens::

    tokeniser_length + 1  START_OF_SPEECH
    tokeniser_length + 2  END_OF_SPEECH
    tokeniser_length + 3  START_OF_HUMAN
    tokeniser_length + 4  END_OF_HUMAN
    tokeniser_length + 5  START_OF_AI
    tokeniser_length + 6  END_OF_AI
    tokeniser_length + 7  PAD

Audio Codec Encoding
--------------------

Audio tokens are interleaved across 4 codebooks. Each audio "frame"
consists of 4 consecutive tokens, one per codebook, offset-encoded::

    token = raw_code + (codebook_index * 4032) + audio_tokens_start

Where ``codebook_index`` is 0..3 and ``audio_tokens_start = tokeniser_length + 10``.

Decoding reverses this: subtract offsets, reshape into [4, num_frames],
and feed into the NeMo codec decoder to produce a PCM waveform at 22 050 Hz.
"""

import torch
import numpy as np
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer

from app.config import Settings


class NemoAudioPlayer:
    """
    Audio processing and playback handler for the Kani TTS model.

    Loads the NVIDIA NeMo Nano Codec on init and provides methods to:
    - Validate model output for required speech markers.
    - Extract and de-interleave audio codec tokens.
    - Decode tokens into a numpy PCM waveform.

    Args:
        config: Application settings (provides tokeniser_length, etc.).
        text_tokenizer_name: Optional HF tokenizer for decoding text from
            the generated sequence (debug / development use only).

    Attributes:
        nemo_codec_model: Pre-trained NeMo Nano Codec (22 kHz, 0.6 kbps).
        device: ``"cuda"`` if a GPU is available, else ``"cpu"``.
        codebook_size: Number of codes per codebook (4032).

    Example::

        player = NemoAudioPlayer(settings)
        waveform, text = player.get_waveform(model_output_tensor)
        # waveform is a float32 numpy array, sample rate 22050
    """

    def __init__(self, config: Settings, text_tokenizer_name: str = None) -> None:
        self.conf = config

        # Load NeMo codec WITHOUT the discriminator to avoid the torch.load
        # CVE-2025-32434 error.  The discriminator is only used during codec
        # training -- encode() and decode() only need encoder + decoder + quantizer.
        _codec_cfg = AudioCodecModel.from_pretrained(
            "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
            return_config=True,
        )
        _codec_cfg.discriminator = None
        self.nemo_codec_model = AudioCodecModel.from_pretrained(
            "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
            override_config_path=_codec_cfg,
            strict=False,
            map_location="cpu",
        )
        self.nemo_codec_model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nemo_codec_model.to(self.device)

        self.text_tokenizer_name = text_tokenizer_name
        if self.text_tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name)

        # Derived token IDs
        self.tokeniser_length = self.conf.tokeniser_length
        self.start_of_text = self.conf.start_of_text
        self.end_of_text = self.conf.end_of_text
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def output_validation(self, out_ids: torch.Tensor) -> None:
        """
        Verify that the generated sequence contains both START_OF_SPEECH
        and END_OF_SPEECH markers.

        Args:
            out_ids: 1-D tensor of token IDs from the model.

        Raises:
            ValueError: If either speech boundary token is missing,
                indicating the model did not produce a valid audio sequence.
        """
        if self.start_of_speech not in out_ids or self.end_of_speech not in out_ids:
            raise ValueError("Special speech tokens not found in output!")

    def get_nano_codes(self, out_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and decode audio codec tokens from the generated sequence.

        Steps:
            1. Locate START_OF_SPEECH and END_OF_SPEECH boundaries.
            2. Slice the audio region.
            3. Reshape into [num_frames, 4] (one column per codebook).
            4. Remove per-codebook offsets to recover raw codec indices.
            5. Transpose to [4, num_frames] for the NeMo decoder.

        Args:
            out_ids: 1-D tensor of token IDs.

        Returns:
            Tuple of (audio_codes, lengths) where:
            - audio_codes: int tensor [1, 4, num_frames]
            - lengths: int tensor [1] containing num_frames

        Raises:
            ValueError: If the audio region is empty, not a multiple of 4,
                or contains invalid (negative) codec indices.
        """
        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError("Invalid audio codes sequence!")

        audio_codes = out_ids[start_a_idx + 1 : end_a_idx]
        if len(audio_codes) % 4:
            raise ValueError("The length of the sequence must be a multiple of 4!")

        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor(
            [self.codebook_size * i for i in range(4)], device=audio_codes.device
        )
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError("Invalid audio tokens!")

        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids: torch.Tensor) -> str:
        """
        Decode the text portion of the generated sequence.

        Extracts tokens between START_OF_TEXT and END_OF_TEXT, then
        decodes them using the HuggingFace tokenizer.

        Args:
            out_ids: 1-D tensor of token IDs.

        Returns:
            Decoded text string.

        Note:
            Requires ``text_tokenizer_name`` to have been provided at init.
        """
        start_t_idx = (out_ids == self.start_of_text).nonzero(as_tuple=True)[0].item()
        end_t_idx = (out_ids == self.end_of_text).nonzero(as_tuple=True)[0].item()
        txt_tokens = out_ids[start_t_idx : end_t_idx + 1]
        return self.tokenizer.decode(txt_tokens, skip_special_tokens=True)

    def get_waveform(self, out_ids: torch.Tensor) -> tuple[np.ndarray, str | None]:
        """
        Convert model output tokens to a playable audio waveform.

        This is the main public method. It validates the output, extracts
        audio codes, decodes them through the NeMo codec, and returns
        a numpy array ready for WAV encoding.

        Args:
            out_ids: Model output tensor (any shape -- will be flattened).

        Returns:
            Tuple of (audio, text) where:
            - audio: float32 numpy array of PCM samples at 22 050 Hz.
            - text: Decoded text string if a text tokenizer was configured,
              otherwise None.

        Raises:
            ValueError: On malformed output (missing markers, bad codec tokens).
        """
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes = audio_codes.to(self.device)
        len_ = len_.to(self.device)

        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(
                tokens=audio_codes, tokens_len=len_
            )
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        text = None
        if self.text_tokenizer_name:
            text = self.get_text(out_ids)

        return output_audio, text
