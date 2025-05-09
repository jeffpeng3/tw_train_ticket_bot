# -*- coding: utf-8 -*-
from pydub import AudioSegment
import librosa
import numpy as np
import os
import tensorflow as tf
from flask import Flask, jsonify, request
import tempfile # 匯入 tempfile 模組
# shutil 不再需要，因為 TemporaryDirectory 會自動清理

# TensorFlow Lite模型初始化
classification = "bcdfghjklmnpqrstvwxy32475689a"
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Failed to load TFLite model: {e}")
    raise

class CaptchaRecognizer:
    def __init__(self):
        pass

    def wav2mfcc(self, file_path, max_pad_len=35):
        try:
            wave, sr = librosa.load(file_path, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(y=wave, sr=16000)
            
            pad_width = max_pad_len - mfcc.shape[1]
            if pad_width < 0:
                # 您提到移除了部分錯誤檢查，這裡保留警告日誌
                print(f"Warning: MFCC sequence length ({mfcc.shape[1]}) > max_pad_len ({max_pad_len}). Truncating.")
                mfcc = mfcc[:, :max_pad_len]
            else:
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
            return mfcc
        except Exception as e:
            print(f"Error in wav2mfcc for {file_path}: {e}")
            raise

    def get_key_part(self, key_part_index, sTime_for_segment, source_audio_path, temp_segment_dir):
        # temp_segment_dir 是 TemporaryDirectory 的路徑
        try:
            song = AudioSegment.from_file(source_audio_path)
            
            startTime_ms = sTime_for_segment * 1000 - 200
            endTime_ms = (sTime_for_segment + 1) * 1000
            
            startTime_ms = max(0, startTime_ms)
            endTime_ms = min(len(song), endTime_ms)

            if startTime_ms >= endTime_ms:
                raise ValueError(f"Invalid time segment for key_part {key_part_index}")

            extract = song[startTime_ms:endTime_ms]
            
            # 在 TemporaryDirectory 管理的目錄中創建片段檔案
            # 檔名可以簡單一點，因為目錄本身是唯一的
            temp_wav_filename = os.path.join(temp_segment_dir, f"kp_{key_part_index}.wav")
            extract.export(temp_wav_filename, format="wav")
            return temp_wav_filename
        except Exception as e:
            print(f"Error in get_key_part for {source_audio_path}, part {key_part_index}: {e}")
            raise

    def predict(self, kp_wav_file_path):
        try:
            mfcc = self.wav2mfcc(str(kp_wav_file_path))
            mfcc_reshaped = mfcc.reshape(1, 20, 35, 1)
            
            interpreter.set_tensor(input_details[0]["index"], mfcc_reshaped)
            interpreter.invoke()
            ans_tensor = interpreter.get_tensor(output_details[0]["index"])
            
            ans_squeezed = np.squeeze(ans_tensor)
            predicted_char_index = np.argmax(ans_squeezed)
            predicted_char = classification[predicted_char_index] 
            
            return predicted_char
        except Exception as e:
            print(f"Error in predict for {kp_wav_file_path}: {e}")
            raise
        # finally 區塊中的 os.remove(kp_wav_file_path) 可以移除
        # 因為 TemporaryDirectory 會在結束時清理所有內容

    def recognize_captcha_from_audio(self, uploaded_audio_path, temp_dir_for_processing):
        captcha_result = ""
        char_time_offsets = [17, 19, 21, 23, 25, 27]
        try:
            for i in range(6):
                sTime = char_time_offsets[i]
                # 片段檔案會創建在 temp_dir_for_processing 中
                kp_file_path = self.get_key_part(
                    i + 1, sTime, uploaded_audio_path, temp_dir_for_processing
                )
                predicted_char = self.predict(kp_file_path)
                captcha_result += predicted_char
            
            return captcha_result
        except Exception as e:
            print(f"Error during captcha recognition for {uploaded_audio_path}: {e}")
            return None

app = Flask(__name__)
captcha_recognizer = CaptchaRecognizer()

@app.route("/captcha", methods=["POST"])
def handle_captcha_request():
    if "audio_file" not in request.files:
        return jsonify(error="No 'audio_file' part in the request"), 400

    file = request.files["audio_file"]

    if not file or file.filename == "":
        return jsonify(error="No selected file or empty filename"), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, file.filename)
            file.save(audio_path)
            result = captcha_recognizer.recognize_captcha_from_audio(
                audio_path, temp_dir
            )

            if result:
                return jsonify(captcha_text=result)
            else:
                return jsonify(error="Failed to recognize captcha from audio"), 500
    except Exception as e:
        # 記錄更詳細的錯誤日誌
        app.logger.error(f"Error processing audio file {file.filename}: {e}", exc_info=True)
        return jsonify(error="Internal server error during audio processing"), 500
    # finally 區塊不再需要手動 shutil.rmtree(temp_dir)，TemporaryDirectory 會自動處理

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
