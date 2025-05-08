import collections
import aiohttp
import librosa
import numpy as np
import soundfile as sf
import subprocess
import os
import torch
from torch import nn, Tensor
import io
from typing import Optional, Any, Union, Type, Coroutine

class CaptchaResolver:
    _instance: Optional["CaptchaResolver"] = None
    _initialized: bool = False

    MODEL_PATH: str
    classification: str
    model: nn.Module

    def __new__(
        cls: Type["CaptchaResolver"], *args: Any, **kwargs: Any
    ) -> "CaptchaResolver":
        if not cls._instance:
            cls._instance = super(CaptchaResolver, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        if CaptchaResolver._initialized:
            return

        self.MODEL_PATH = "model.pt"
        self.classification = "bcdfghjklmnpqrstvwxy32475689a"
        self.ffmpeg_path = "ffmpeg"
        self.model = torch.load(self.MODEL_PATH, weights_only=False)
        self.model.eval()
        CaptchaResolver._initialized = True

    def _wav2mfcc(
        self, audio_input: Union[str, io.BytesIO], max_pad_len: int = 35
    ) -> Optional[np.ndarray]:
        """Converts WAV audio (from file-like object or path) to MFCC features."""
        try:
            wave, sr = librosa.load(audio_input, mono=True, sr=None)
        except Exception as e:
            return None

        if wave is None or len(wave) == 0:
            return None

        wave = wave[::3]

        if len(wave) == 0:
            return None

        n_fft_to_use = 2048
        if len(wave) < n_fft_to_use:
            n_fft_to_use = len(wave)
        
        if n_fft_to_use < 4:
            return None

        mfcc: np.ndarray = librosa.feature.mfcc(y=wave, sr=16000, n_fft=n_fft_to_use)

        if mfcc.shape[1] < max_pad_len:
            pad_width: int = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        elif mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    def _predict_captcha_batch(self, mfcc_batch_tensor: Tensor) -> list[str]:
        """Predicts characters from a batch of MFCC tensors using PyTorch."""
        with torch.no_grad():
            predictions: Tensor = self.model(mfcc_batch_tensor)

        predictions_np: np.ndarray = predictions.cpu().numpy()
        predicted_indices: np.ndarray = np.argmax(predictions_np, axis=1)

        predicted_chars: list[str] = []
        for idx in predicted_indices:
            if idx < len(self.classification):
                predicted_chars.append(self.classification[idx])
            else:
                raise ValueError(
                    f"Predicted index {idx} out of bounds for classification string."
                )

        return "".join(predicted_chars)

    def _get_char_from_raw_prediction_segment(self, segment: str) -> str:
        counts = collections.Counter(segment)
        most_common_char = counts.most_common(1)[0][0]
        return most_common_char

    def _process_audio_to_text(self, audio_stream: io.BufferedIOBase, trans = True) -> str:
        y: Optional[np.ndarray] = None
        sr: int = 0

        input_bytes: Optional[bytes] = None
        if hasattr(audio_stream, "read"):
            input_bytes = audio_stream.read()
            if not isinstance(input_bytes, bytes):
                print("ERROR: Audio stream read did not return bytes.")
                return ""
        
        ffmpeg_cmd_list = []
        if trans:
            ffmpeg_cmd_list = [
                self.ffmpeg_path,
                "-ss", "00:00:16.6",
                "-i", "-",
                "-f", "wav",
                "-to", "00:00:11",
                "-hide_banner", "-loglevel", "error",
                "-"
            ]
        else:
            ffmpeg_cmd_list = [
                self.ffmpeg_path,
                "-i", "-",
                "-f", "wav",
                "-hide_banner", "-loglevel", "error",
                "-"
            ]
        
        process_result = subprocess.run(
            ffmpeg_cmd_list, input=input_bytes, capture_output=True, check=False
        )


        if process_result.returncode != 0:
            print(f"ERROR: ffmpeg processing failed. Stderr: {process_result.stderr.decode('utf-8', errors='ignore')}")
            return ""

        audio_bytes_io = io.BytesIO(process_result.stdout)
        if not process_result.stdout:
            print("ERROR: ffmpeg produced no output.")
            return ""
            
        try:
            y_arr, sr_float = librosa.load(audio_bytes_io, sr=None)
            y = y_arr
            sr = int(sr_float)
        except Exception as e:
            print(f"ERROR: librosa.load failed after ffmpeg. Error: {e}")
            return ""

        if y is None or len(y) == 0:
            print("ERROR: Audio data is empty after librosa.load.")
            return ""

        all_mfccs_list: list[np.ndarray] = []
        window_duration_ms = 1300
        step_ms = 50
        audio_duration_ms = len(y) / sr * 1000

        
        num_segments = 0
        for startTime_ms in range(0, int(audio_duration_ms) + 1, step_ms):
            endTime_ms: int = startTime_ms + window_duration_ms

            start_sample: int = int(startTime_ms  * sr / 1000)
            end_sample: int = int(endTime_ms * sr / 1000 )

            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            
            if start_sample >= end_sample:
                continue

            segment: np.ndarray = y[start_sample:end_sample]

            bytes_io_obj: io.BytesIO = io.BytesIO()
            sf.write(bytes_io_obj, segment, sr, format="WAV", subtype="PCM_16")
            bytes_io_obj.seek(0)

            mfcc: Optional[np.ndarray] = self._wav2mfcc(bytes_io_obj)
            if mfcc is not None:
                 all_mfccs_list.append(mfcc)
                 num_segments += 1
            else:
                print(f"Warning: MFCC for segment starting at {startTime_ms}ms was None.")
        
        if not all_mfccs_list:
            print("ERROR: No MFCC features could be extracted.")
            return ""

        L_raw: int = num_segments

        final_captcha_chars: list[str] = []
        intervals: list[int] = [1, 42, 39, 41, 37, 40]
        core_sample_radius: int = 3
        user_defined_center_points = [1, 43, 83, 125, 165, 204]
        
        all_calculated_core_indices = []
        max_required_idx = -1
        _current_cumulative_offset_for_calc = 0


        for idx, _interval_len_for_calc in enumerate(intervals):
            _actual_center_idx = user_defined_center_points[idx]

            _start_core_idx_calc = max(0, _actual_center_idx - core_sample_radius)
            _end_core_idx_calc = _actual_center_idx + core_sample_radius + 1
            
            all_calculated_core_indices.append((_start_core_idx_calc, _end_core_idx_calc))

            if _end_core_idx_calc > _start_core_idx_calc:
                 max_required_idx = max(max_required_idx, _end_core_idx_calc - 1)

            _current_cumulative_offset_for_calc += _interval_len_for_calc
        


        for i, interval_len_loop in enumerate(intervals):
            start_core_idx, end_core_idx = all_calculated_core_indices[i]

            if start_core_idx >= L_raw:
                print(f"Warning: Char {i+1}: Start index {start_core_idx} for MFCC segment is out of bounds (L_raw={L_raw}). Appending '?'")
                final_captcha_chars.append("?")
                continue

            actual_end_core_idx = min(end_core_idx, L_raw)

            predicted_char: str
            if start_core_idx >= actual_end_core_idx:
                print(f"Warning: Char {i+1}: Core MFCC segment is invalid or empty. Indices for all_mfccs_list: [{start_core_idx}:{actual_end_core_idx}], L_raw={L_raw}. Appending '?'")
                predicted_char = "?"
            else:
                core_segment_mfccs = all_mfccs_list[start_core_idx:actual_end_core_idx]

                if not core_segment_mfccs:
                    print(f"Warning: Char {i+1}: Extracted core_segment_mfccs is empty for indices [{start_core_idx}:{actual_end_core_idx}]. Appending '?'")
                    predicted_char = "?"
                else:
                    try:
                        stacked_core_mfccs = np.stack(core_segment_mfccs)
                        mfcc_batch_tensor_for_segment = torch.from_numpy(stacked_core_mfccs).float().unsqueeze(1)
                        
                        raw_chars_for_segment: str = self._predict_captcha_batch(mfcc_batch_tensor_for_segment)

                        predicted_char = self._get_char_from_raw_prediction_segment(raw_chars_for_segment)
                        if not predicted_char:
                            print(f"Warning: Char {i+1}: _get_char_from_raw_prediction_segment returned empty for raw_chars '{raw_chars_for_segment}'. Appending '?'")
                            predicted_char = "?"
                    except Exception as e_pred:
                        print(f"ERROR: Char {i+1}: Failed during prediction for MFCC segment [{start_core_idx}:{actual_end_core_idx}]. Error: {e_pred}. Appending '?'")
                        predicted_char = "?"
            final_captcha_chars.append(predicted_char)

        final_captcha: str = "".join(final_captcha_chars)
        return final_captcha

    async def resolve_audio_captcha(
        self, session: aiohttp.ClientSession, audio_source: str
    ) -> Coroutine[Any, Any, str]:
        """
        Downloads or reads, processes, and predicts captcha from an audio URL or local file path asynchronously.
        Returns the 6-character captcha string, or an empty string if failed.
        """
        audio_content: Optional[bytes] = None
        if audio_source.startswith(("http://", "https://")):
            async with session.get(audio_source) as audio_response:
                audio_response.raise_for_status()
                audio_content = await audio_response.read()
        elif os.path.exists(audio_source):
            with open(audio_source, "rb") as f:
                audio_content = f.read()
        else:
            raise ValueError(
                f"Invalid audio source: '{audio_source}'. Must be a URL or a valid file path."
            )
        audio_stream: io.BytesIO = io.BytesIO(audio_content)
        captcha_text: str = self._process_audio_to_text(audio_stream)
        return captcha_text
