# -*- coding: utf-8 -*-
import aiohttp
import librosa
import numpy as np
import soundfile as sf
import subprocess
import os
import shutil
import torch
from torch import nn, Tensor
import io
from typing import Optional, Any, Union, Type, Coroutine


class CaptchaResolver:
    _instance: Optional["CaptchaResolver"] = None
    _initialized: bool = False

    MODEL_PATH: str
    classification: str
    model: nn.Module  # type: ignore

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
        self.ffmpeg_path = "ffmpeg"  # Added ffmpeg_path
        self.model = torch.load(self.MODEL_PATH, weights_only=False)  # type: ignore
        self.model.eval()
        CaptchaResolver._initialized = True

    def _wav2mfcc(
        self, audio_input: Union[str, io.BytesIO], max_pad_len: int = 35
    ) -> Optional[np.ndarray]:
        """Converts WAV audio (from file-like object or path) to MFCC features."""
        wave, sr = librosa.load(audio_input, mono=True, sr=None)  # type: ignore
        wave = wave[::3]
        mfcc: np.ndarray = librosa.feature.mfcc(y=wave, sr=16000)  # type: ignore

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

    def _is_ffmpeg_available(self) -> bool:
        return shutil.which(self.ffmpeg_path) is not None

    def _process_audio_to_text(self, audio_stream: io.BufferedIOBase) -> str:
        y: Optional[np.ndarray] = None
        sr: int = 0

        # Determine minimum required duration (approx 28.2 seconds for 6 chars starting at 17s, 2s apart, 1.2s slice)
        MIN_REQUIRED_DURATION_SEC = (17 + (6 - 1) * 2) + 1.2

        if not self._is_ffmpeg_available():
            print(
                f"ERROR: ffmpeg not found at path: {self.ffmpeg_path}. Cannot process audio."
            )
            return ""

        input_bytes: Optional[bytes] = None
        try:
            if hasattr(audio_stream, "read"):
                input_bytes = audio_stream.read()
                if not isinstance(input_bytes, bytes):  # Ensure it's bytes
                    print("ERROR: Audio stream read did not return bytes.")
                    return ""
            else:
                print(
                    "ERROR: Audio stream is not readable or does not have a read method."
                )
                return ""
        except Exception as e_read:
            print(f"ERROR: Error reading audio stream: {e_read}")
            return ""

        if not input_bytes:
            print("ERROR: Audio stream was empty or could not be read.")
            return ""

        try:
            ffmpeg_cmd = [
                self.ffmpeg_path,
                "-i",
                "-",  # Input from stdin
                "-f",
                "wav",
                "-hide_banner",
                "-loglevel",
                "error",
                "-",  # Output to stdout
            ]
            process_result = subprocess.run(
                ffmpeg_cmd, input=input_bytes, capture_output=True, check=False
            )

            if process_result.returncode == 0:
                audio_bytes_io = io.BytesIO(process_result.stdout)
                y_arr, sr_float = librosa.load(audio_bytes_io, sr=None, duration=30.0)  # type: ignore
                y = y_arr
                sr = int(sr_float)
            else:
                print(
                    f"ERROR: ffmpeg processing failed. Return code: {process_result.returncode}"
                )
                stderr_output = (
                    process_result.stderr.decode("utf-8", errors="ignore")
                    if process_result.stderr
                    else "No stderr output"
                )
                print(f"ERROR: ffmpeg stderr: {stderr_output}")
                # y remains None
        except FileNotFoundError:
            print(
                f"ERROR: ffmpeg executable not found at '{self.ffmpeg_path}'. Please ensure ffmpeg is installed and in PATH or path is configured correctly."
            )
            return ""
        except Exception as e_ffmpeg:
            print(
                f"ERROR: Error during ffmpeg processing or librosa loading: {e_ffmpeg}"
            )
            # y remains None

        if y is None or sr == 0:
            print(
                "ERROR: Failed to load audio data using ffmpeg from the provided stream."
            )
            return ""

        loaded_duration_sec = len(y) / sr

        if loaded_duration_sec < MIN_REQUIRED_DURATION_SEC:
            print(
                f"ERROR: Audio duration ({loaded_duration_sec:.2f}s) is insufficient. Required: {MIN_REQUIRED_DURATION_SEC:.2f}s."
            )
            return ""

        mfcc_list: list[np.ndarray] = []
        num_chars: int = 6
        try:
            for i in range(num_chars):
                start_time_sec: int = 17 + i * 2
                startTime_ms: int = start_time_sec * 1000 - 200
                endTime_ms: int = (start_time_sec + 1) * 1000

                start_sample: int = int(startTime_ms / 1000 * sr)
                end_sample: int = int(endTime_ms / 1000 * sr)

                # Ensure samples are within bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(y), end_sample)

                if start_sample >= end_sample:
                    print(
                        f"ERROR: Segment {i + 1} has invalid time range after conversion: start_sample={start_sample}, end_sample={end_sample}"
                    )
                    print(
                        f"ERROR: Error: Segment {i + 1} results in empty or invalid sample range."
                    )
                    return ""

                segment: np.ndarray = y[start_sample:end_sample]

                bytes_io_obj: io.BytesIO = io.BytesIO()
                sf.write(bytes_io_obj, segment, sr, format="WAV", subtype="PCM_16")
                bytes_io_obj.seek(0)

                mfcc: Optional[np.ndarray] = self._wav2mfcc(bytes_io_obj)
                if mfcc is None:
                    print(f"ERROR: MFCC extraction failed for segment {i + 1}.")
                    return ""
                mfcc_list.append(mfcc)

            if len(mfcc_list) != num_chars:
                print(
                    f"ERROR: Expected {num_chars} MFCC segments, but got {len(mfcc_list)}"
                )
                return ""

            stacked_mfccs: np.ndarray = np.stack(mfcc_list)  # type: ignore
            mfcc_batch_tensor: Tensor = (
                torch.from_numpy(stacked_mfccs).float().unsqueeze(1)
            )

            ans: str = self._predict_captcha_batch(mfcc_batch_tensor)

            if "?" in ans:
                return ""

        except Exception as e:
            print(f"ERROR: Error processing audio for batch prediction: {e}")
            return ""

        return ans

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
        elif os.path.exists(audio_source):  # Check if it's an existing file path
            with open(audio_source, "rb") as f:
                audio_content = f.read()
        else:
            raise ValueError(
                f"Invalid audio source: '{audio_source}'. Must be a URL or a valid file path."
            )

        audio_stream: io.BytesIO = io.BytesIO(audio_content)
        captcha_text: str = self._process_audio_to_text(audio_stream)
        return captcha_text
