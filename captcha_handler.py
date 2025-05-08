# -*- coding: utf-8 -*-
import collections # 新增 collections 用於投票
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
from time import time

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

    def _get_char_from_raw_prediction_segment(self, segment: str) -> str:
        """
        從原始預測片段中獲取最可能的單個字符。
        使用投票法，如果票數相同，則選擇第一個出現的字符。
        """
        if not segment:
            return ""  # 或其他標識符，例如 "?"
        
        # 投票選出最常見的字符
        counts = collections.Counter(segment)
        # 找到最高票數
        max_count = 0
        for char in counts:
            if counts[char] > max_count:
                max_count = counts[char]
        
        # 找出所有最高票數的字符 (可能有多個)
        # 為了穩定性，我們選擇在原始 segment 中第一個出現的最高票字符
        # 或者，可以簡單地選擇 counts.most_common(1)[0][0]
        # 這裡我們選擇 most_common，它會處理平局情況（雖然不保證順序）
        # 為了更精確地符合「選擇第一個出現」，需要更複雜的邏輯，但通常 most_common(1) 夠用
        if not counts: # 再次檢查，雖然前面有 if not segment
            return ""

        most_common_char = counts.most_common(1)[0][0]
        # print(f"    Segment '{segment}' -> Most common: '{most_common_char}' (Counts: {counts})") # 日誌
        return most_common_char

    def _process_audio_to_text(self, audio_stream: io.BufferedIOBase, trans = True) -> str:
        y: Optional[np.ndarray] = None
        sr: int = 0

        input_bytes: Optional[bytes] = None
        if hasattr(audio_stream, "read"):
            input_bytes = audio_stream.read()
            if not isinstance(input_bytes, bytes):  # Ensure it's bytes
                print("ERROR: Audio stream read did not return bytes.")
                return ""
        
        ffmpeg_cmd_list = [] # 使用列表以避免在 if/else 中重複
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

        # 移除舊的音頻保存邏輯，除非調試時需要
        # if trans:
        #     with open(f"audio/{int(time())}.wav", "wb") as f:
        #         f.write(process_result.stdout)

        if process_result.returncode != 0:
            print(f"ERROR: ffmpeg processing failed. Stderr: {process_result.stderr.decode('utf-8', errors='ignore')}")
            return ""

        audio_bytes_io = io.BytesIO(process_result.stdout)
        if not process_result.stdout: # 檢查 ffmpeg 是否產生了輸出
            print("ERROR: ffmpeg produced no output.")
            return ""
            
        try:
            y_arr, sr_float = librosa.load(audio_bytes_io, sr=None)  # type: ignore
            y = y_arr
            sr = int(sr_float)
        except Exception as e:
            print(f"ERROR: librosa.load failed after ffmpeg. Error: {e}")
            # 可以考慮保存 ffmpeg 的輸出以供調試
            # with open("ffmpeg_output_error.wav", "wb") as f_err:
            #     f_err.write(process_result.stdout)
            return ""

        if y is None or len(y) == 0:
            print("ERROR: Audio data is empty after librosa.load.")
            return ""

        mfcc_list: list[np.ndarray] = []
        # 滑動窗口參數 (保持與原碼一致)
        window_duration_ms = 1300
        step_ms = 50
        # 根據 ffmpeg 處理後的音頻長度（約11秒）調整循環範圍
        # ffmpeg -to 00:00:11 表示音頻長度上限為 11000 ms
        # 原始循環是 range(0, 10551, 50)
        # 這裡我們假設 y 的長度對應於 ffmpeg 處理後的音頻
        audio_duration_ms = len(y) / sr * 1000

        # for startTime_ms in range(0, 10551, 50): # 舊的固定範圍
        # 根據實際音頻長度動態調整，確保 endTime_ms 不會遠超 audio_duration_ms
        # 這裡的循環條件需要仔細考慮，以匹配模型訓練時的特徵提取方式
        # 原始代碼的 10551 可能是基於特定音頻長度計算的，這裡我們保持它，但要注意如果音頻更短會怎樣
        # 實際上，模型期望的輸入序列長度是固定的（由 _wav2mfcc 中的 max_pad_len 暗示）
        # 滑動窗口的數量決定了 raw_model_output 的長度
        
        num_segments = 0
        for startTime_ms in range(0, int(audio_duration_ms) + 1, step_ms):
            endTime_ms: int = startTime_ms + window_duration_ms

            start_sample: int = int(startTime_ms  * sr / 1000)
            end_sample: int = int(endTime_ms * sr / 1000 )

            # 邊界檢查 (雖然 librosa.load 後 y 的長度是確定的，但以防萬一)
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            
            if start_sample >= end_sample: # 如果片段無效則跳過
                continue

            segment: np.ndarray = y[start_sample:end_sample]

            bytes_io_obj: io.BytesIO = io.BytesIO()
            sf.write(bytes_io_obj, segment, sr, format="WAV", subtype="PCM_16")
            bytes_io_obj.seek(0)

            mfcc: Optional[np.ndarray] = self._wav2mfcc(bytes_io_obj) # max_pad_len=35
            if mfcc is not None:
                 mfcc_list.append(mfcc)
                 num_segments += 1
            else:
                print(f"Warning: MFCC for segment starting at {startTime_ms}ms was None.")
        
        if not mfcc_list:
            print("ERROR: No MFCC features could be extracted.")
            return ""

        stacked_mfccs: np.ndarray = np.stack(mfcc_list)
        mfcc_batch_tensor: Tensor = (
            torch.from_numpy(stacked_mfccs).float().unsqueeze(1)
        )

        raw_model_output: str = self._predict_captcha_batch(mfcc_batch_tensor)
        print(f"Log: Raw model output: '{raw_model_output}' (Length: {len(raw_model_output)})")

        # --- 新的基於固定間隔長度的核心採樣邏輯 ---
        final_captcha_chars: list[str] = []
        intervals: list[int] = [1, 42, 39, 40, 38, 40]  # 使用者提供的固定間隔長度
        
        L_raw: int = len(raw_model_output)
        sum_intervals: int = sum(intervals)

        # 日誌記錄 raw_model_output 的總長度和 sum_intervals
        print(f"Log: Total length of raw_model_output: {L_raw}")
        print(f"Log: Sum of specified intervals: {sum_intervals}")

        if L_raw == 0:
            print("ERROR: Raw model output is empty. Cannot proceed.")
            return ""

        # 主要檢查：raw_model_output 的總長度是否大於或等於 sum_intervals
        if L_raw < sum_intervals:
            print(f"ERROR: Raw model output length ({L_raw}) is less than the sum of intervals ({sum_intervals}). Cannot reliably extract all characters.")
            return ""  # 根據指示，長度不足時返回空字串

        current_cumulative_offset: int = 0
        core_sample_radius: int = 3  # 中心點前後各取3個點，總共 2*3+1 = 7點

        for i, interval_len in enumerate(intervals):
            # 計算此字元在其自身間隔內的相對中心點
            relative_center: int = interval_len // 2
            # 計算此字元在整個 raw_model_output 中的絕對中心點索引
            center_idx: int = current_cumulative_offset + relative_center

            # 提取核心採樣片段的起始和結束索引
            # 嚴格處理邊界條件
            # center_idx is calculated based on interval_len // 2 for logging and i==0 case
            # (This variable should already be defined from line 252: center_idx = current_cumulative_offset + (interval_len // 2) )

            if i == 0:
                # For the first character, use the original logic based on its interval's midpoint (center_idx)
                start_core_idx = max(0, center_idx - core_sample_radius)
                end_core_idx = min(L_raw, center_idx + core_sample_radius + 1)
            else:
                # For subsequent characters (i > 0)
                # The sampling window starts at the last index of the current character's segment
                _start_offset_in_raw_output = current_cumulative_offset + interval_len - 1
                start_core_idx = max(0, _start_offset_in_raw_output) # Ensure non-negative
                
                window_len = (2 * core_sample_radius) + 1 # Should be 7 if core_sample_radius is 3
                end_core_idx = min(L_raw, start_core_idx + window_len)
            
            core_segment: str
            # 確保提取的片段不為空 (start_core_idx < end_core_idx)
            if start_core_idx >= end_core_idx:
                print(f"Warning: Char {i+1}: Core segment calculation resulted in empty or invalid range. interval_len={interval_len}, center_idx={center_idx}, calculated_indices=[{start_core_idx}:{end_core_idx}]. Appending '?'")
                core_segment = ""
            else:
                core_segment = raw_model_output[start_core_idx:end_core_idx]

            # 日誌記錄每個字元的處理信息
            print(f"Log: Char {i+1}: interval_len={interval_len}, current_cumulative_offset={current_cumulative_offset}, center_idx={center_idx}, Core_indices=[{start_core_idx}:{end_core_idx}], Core_segment='{core_segment}'")

            predicted_char: str = self._get_char_from_raw_prediction_segment(core_segment)
            
            if not predicted_char: # 如果投票結果為空
                print(f"Warning: Char {i+1}: _get_char_from_raw_prediction_segment returned empty for segment '{core_segment}'. Appending '?'")
                final_captcha_chars.append("?") # 使用 '?' 作為錯誤標識
            else:
                final_captcha_chars.append(predicted_char)
            print(f"Log: Char {i+1} predicted: '{predicted_char}'")
            
            # 更新 current_cumulative_offset 為下一個間隔的開始做準備
            current_cumulative_offset += interval_len

        final_captcha: str = "".join(final_captcha_chars)
        print(f"Log: Final predicted captcha (using fixed intervals): '{final_captcha}'")
        
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
