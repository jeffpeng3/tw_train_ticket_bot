# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import requests
from pydub import AudioSegment
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import io

class CaptchaResolver:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CaptchaResolver, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if CaptchaResolver._initialized:
            return
        
        self.MODEL_PATH = "model.tflite"
        self.classification = [
            "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
            "n", "p", "q", "r", "s", "t", "v", "w", "x", "y",
            "3", "2", "4", "7", "5", "6", "8", "9", "a",
        ] # Maintained original classification list
        self.interpreter = tf.lite.Interpreter(model_path=self.MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        CaptchaResolver._initialized = True

    def _wav2mfcc(self, audio_input, max_pad_len=35):
        """Converts WAV audio (from file-like object or path) to MFCC features."""
        try:
            # librosa.load can handle file-like objects directly
            wave, sr = librosa.load(audio_input, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(y=wave, sr=16000)
            pad_width = max_pad_len - mfcc.shape[1]
            if pad_width < 0:
                mfcc = mfcc[:, :max_pad_len]
                pad_width = 0
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
            return mfcc
        except Exception as e:
            print(f"Error processing wav audio data: {e}")
            return None

    def _predict_captcha_char(self, audio_segment_bytes_io):
        """Predicts a single character from an audio segment (BytesIO)."""
        mfcc = self._wav2mfcc(audio_segment_bytes_io)
        if mfcc is None:
            return None
        mfcc_reshaped = mfcc.reshape(1, 20, 35, 1).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], mfcc_reshaped)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]["index"])
        prediction = np.squeeze(prediction)
        predicted_char = self.classification[np.argmax(prediction)]
        return predicted_char

    def _process_audio_to_text(self, main_audio_file_path):
        """Processes the main audio MP3 file to extract captcha text."""
        ans = ""
        try:
            song = AudioSegment.from_mp3(main_audio_file_path)
        except Exception as e:
            print(f"Error loading MP3 file {main_audio_file_path}: {e}")
            return "" # Return empty string on error as per original logic
            
        try:
            for i in range(6): # Assuming 6 characters in captcha
                start_time_sec = 17 + i * 2
                startTime_ms = start_time_sec * 1000 - 200
                endTime_ms = (start_time_sec + 1) * 1000
                current_segment = song[startTime_ms:endTime_ms]

                # Create a BytesIO object to store WAV data in memory
                bytes_io_obj = io.BytesIO()
                current_segment.export(out_f=bytes_io_obj, format="wav")
                bytes_io_obj.seek(0) # Reset stream position to the beginning

                predicted_char = self._predict_captcha_char(bytes_io_obj)
                if predicted_char == "?": # Original code had a check for "?", assuming it's an error/unknown
                    # print(f"Warning: Model predicted '?' for character {i+1}. Stopping.") # Optional: more verbose logging
                    return "" # Return empty string as per original logic
                if predicted_char:
                    ans += predicted_char
                else:
                    # _predict_captcha_char would have printed an error
                    return "" # Return empty string on error
        except Exception as e:
            print(f"Error processing character {i + 1} from audio: {e}")
            return "" # Return empty string on error

        return ans

    def resolve_audio_captcha(self, session: requests.Session, audio_captcha_url: str) -> str:
        """
        Downloads, processes, and predicts captcha from an audio URL.
        Returns the 6-character captcha string, or an empty string if failed.
        """
        audio_response = session.get(audio_captcha_url, stream=True)
        audio_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Use a temporary file to store the downloaded MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp_mp3_file:
            try:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    tmp_mp3_file.write(chunk)
                tmp_mp3_file.flush() # Ensure all data is written to disk before passing the name
                captcha_text = self._process_audio_to_text(tmp_mp3_file.name)
                return captcha_text
            except Exception as e:
                # This catches errors during file writing or _process_audio_to_text
                print(f"Error processing temporary audio file or predicting text: {e}")
                return ""

# To use this class:
# from captcha_handler import CaptchaResolver
# captcha_solver = CaptchaResolver()
# import requests
# session = requests.Session()
# captcha_url = "YOUR_ACTUAL_CAPTCHA_AUDIO_URL_HERE"
# result = captcha_solver.resolve_audio_captcha(session, captcha_url)
# print(f"Predicted Captcha: {result}")
