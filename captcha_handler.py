# -*- coding: utf-8 -*-
import aiohttp
from pydub import AudioSegment
import librosa
import numpy as np
import torch
from torch import nn, Tensor
import io
from typing import Optional, Any, Union, Type, Coroutine

class CaptchaResolver:
    _instance: Optional['CaptchaResolver'] = None
    _initialized: bool = False

    MODEL_PATH: str
    classification: str
    model: nn.Module # type: ignore

    def __new__(cls: Type['CaptchaResolver'], *args: Any, **kwargs: Any) -> 'CaptchaResolver':
        if not cls._instance:
            cls._instance = super(CaptchaResolver, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        if CaptchaResolver._initialized:
            return
        
        self.MODEL_PATH = "model.pt"
        self.classification = "bcdfghjklmnpqrstvwxy32475689a"
        self.model = torch.load(self.MODEL_PATH, weights_only=False) # type: ignore
        self.model.eval()
        CaptchaResolver._initialized = True

    def _wav2mfcc(self, audio_input: Union[str, io.BytesIO], max_pad_len: int = 35) -> Optional[np.ndarray]:
        """Converts WAV audio (from file-like object or path) to MFCC features."""
        try:
            wave: np.ndarray
            sr: float
            wave, sr = librosa.load(audio_input, mono=True, sr=None) # type: ignore
            wave = wave[::3]
            mfcc: np.ndarray = librosa.feature.mfcc(y=wave, sr=16000) # type: ignore
            
            if mfcc.shape[1] < max_pad_len:
                pad_width: int = max_pad_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            elif mfcc.shape[1] > max_pad_len:
                mfcc = mfcc[:, :max_pad_len]
            
            return mfcc
        except Exception as e:
            print(f"Error processing wav audio data: {e}")
            return None

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
                print(f"Predicted index {idx} out of bounds for classification list.")
                predicted_chars.append("?")
        
        return predicted_chars

    def _process_audio_to_text(self, main_audio_file_path: Union[str, io.BytesIO]) -> str:
        """Processes the main audio MP3 file to extract captcha text using batch inference."""
        try:
            song: AudioSegment = AudioSegment.from_mp3(main_audio_file_path) # type: ignore
        except Exception as e:
            print(f"Error loading MP3 file {main_audio_file_path}: {e}")
            return ""

        mfcc_list: list[np.ndarray] = []
        num_chars: int = 6
        try:
            for i in range(num_chars):
                start_time_sec: int = 17 + i * 2
                startTime_ms: int = start_time_sec * 1000 - 200
                endTime_ms: int = (start_time_sec + 1) * 1000
                current_segment: AudioSegment = song[startTime_ms:endTime_ms]

                bytes_io_obj: io.BytesIO = io.BytesIO()
                current_segment.export(out_f=bytes_io_obj, format="wav")
                bytes_io_obj.seek(0)

                mfcc: Optional[np.ndarray] = self._wav2mfcc(bytes_io_obj)
                if mfcc is None:
                    print(f"MFCC extraction failed for segment {i+1}.")
                    return ""
                mfcc_list.append(mfcc)
            
            if len(mfcc_list) != num_chars:
                print(f"Expected {num_chars} MFCC segments, but got {len(mfcc_list)}")
                return ""

            stacked_mfccs: np.ndarray = np.stack(mfcc_list) # type: ignore
            mfcc_batch_tensor: Tensor = torch.from_numpy(stacked_mfccs).float().unsqueeze(1)

            predicted_chars_list: list[str] = self._predict_captcha_batch(mfcc_batch_tensor)

            # The check for `predicted_chars_list is None` is removed as _predict_captcha_batch
            # is typed to always return list[str]. If it could return None,
            # its return type hint should be Optional[list[str]].
            # if predicted_chars_list is None:
            #     return ""

            ans: str = "".join(predicted_chars_list)
            if "?" in ans:
                return ""

        except Exception as e:
            print(f"Error processing audio for batch prediction: {e}")
            return ""

        return ans

    async def resolve_audio_captcha(self, session: aiohttp.ClientSession, audio_captcha_url: str) -> Coroutine[Any, Any, str]:
        """
        Downloads, processes, and predicts captcha from an audio URL asynchronously.
        Returns the 6-character captcha string, or an empty string if failed.
        """
        try:
            audio_response: aiohttp.ClientResponse
            async with session.get(audio_captcha_url) as audio_response:
                audio_response.raise_for_status()
                audio_content: bytes = await audio_response.read()
            
            # Use io.BytesIO to handle the audio content in memory
            audio_stream: io.BytesIO = io.BytesIO(audio_content)
            with audio_stream: # No need to assign to a new variable if just using it in with context
                captcha_text: str = self._process_audio_to_text(audio_stream) # _process_audio_to_text expects a file-like object or path
                return captcha_text
        except aiohttp.ClientError as e:
            print(f"Error downloading audio captcha: {e}")
            return ""
        except Exception as e:
            print(f"Error processing audio or predicting text: {e}")
            return ""

