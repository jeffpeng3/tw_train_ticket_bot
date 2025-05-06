# -*- coding: utf-8 -*-
import requests
from pydub import AudioSegment
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import io

MODEL_PATH = "model.tflite"

classification = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', '3', '2', '4', '7', '5', '6', '8', '9', 'a']
interpreter = None
input_details = None
output_details = None

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def _wav2mfcc(audio_input, max_pad_len=35): # <--- 修改參數名稱 file_path 為 audio_input
    """Converts WAV audio (from file-like object or path) to MFCC features."""
    try:
        # librosa.load可以直接處理類檔案物件
        wave, sr = librosa.load(audio_input, mono=True, sr=None)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(y=wave, sr=16000)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width < 0:
            mfcc = mfcc[:, :max_pad_len]
            pad_width = 0
            # print(f"Warning: MFCC features truncated") # Less verbose, path no longer relevant here
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfcc
    except Exception as e:
        # 移除 file_path，因為可能是 BytesIO
        print(f"Error processing wav audio data: {e}")
        return None

def _predict_captcha_char(audio_segment_bytes_io): # <--- 修改參數以接受 BytesIO 物件
    """Predicts a single character from an audio segment (BytesIO)."""
    # 模組級別已處理 interpreter, input_details, output_details 的載入失敗
    # 如果執行到此處，它們必然已被成功初始化
    try:
        mfcc = _wav2mfcc(audio_segment_bytes_io) # <--- 傳遞 BytesIO 物件
        if mfcc is None:
            return None
        mfcc_reshaped = mfcc.reshape(1, 20, 35, 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], mfcc_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(prediction)
        predicted_char = classification[np.argmax(prediction)]
        return predicted_char
    except Exception as e:
        # 移除 kp_file_path，因為現在是 BytesIO
        print(f"Error predicting captcha char from audio data: {e}")
        return None

def _process_audio_to_text(main_audio_file_path):
    """Processes the main audio MP3 file to extract captcha text."""
    ans = ""
    song = AudioSegment.from_mp3(main_audio_file_path)
    for i in range(6):
        try:
            start_time_sec = 17 + i * 2
            startTime_ms = start_time_sec * 1000 - 200
            endTime_ms = (start_time_sec + 1) * 1000
            current_segment = song[startTime_ms:endTime_ms]

            # 創建一個 BytesIO 物件來儲存 WAV 數據
            bytes_io_obj = io.BytesIO()
            current_segment.export(out_f=bytes_io_obj, format="wav")
            bytes_io_obj.seek(0) # 重置指標到開頭，以便 librosa.load 讀取

            predicted_char = _predict_captcha_char(bytes_io_obj) # <--- 傳遞 BytesIO 物件
            if predicted_char == "?":
                return "" # 當任一音訊片段預測為 '?' 時，立即停止處理並返回空字串
            if predicted_char:
                ans += predicted_char
            else:
                # 如果 _predict_captcha_char 返回 None (表示預測失敗)
                # 也視為辨識失敗，立即返回空字串
                return ""
            # 不需要再關閉 bytes_io_obj，它會在作用域結束時自動清理
        except Exception as e:
            print(f"Error processing character {i+1} from audio: {e}")
            # 發生例外也立即返回空字串
            return ""

    # 只有當迴圈正常完成且 ans 長度為 6 (意味著沒有提早退出)
    # 才返回 ans。若迴圈因任何原因提早退出，此處不會執行。
    return ans

def resolve_audio_captcha(session: requests.Session, audio_captcha_url: str) -> str:
    """
    Downloads, processes, and predicts captcha from an audio URL.
    Returns the 6-character captcha string, or an empty string if failed.
    """
    audio_response = session.get(audio_captcha_url, stream=True)
    audio_response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_mp3_file:
        for chunk in audio_response.iter_content(chunk_size=8192):
            tmp_mp3_file.write(chunk)
        captcha_text = _process_audio_to_text(tmp_mp3_file.name)
        return captcha_text