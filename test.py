import requests
import os
import time

# API 端點 URL
url = "http://localhost:8000/captcha"

# 音訊檔案路徑 (假設 audio.mp3 與 test.py 在同一目錄)
audio_file_path = "audio.mp3"

# 要呼叫 API 的次數
num_requests = 5

def test_captcha_api_multiple_times(file_path, count):
    """
    多次測試驗證碼 API 並計時。

    :param file_path: 要上傳的音訊檔案路徑。
    :param count: 呼叫 API 的次數。
    """
    if not os.path.exists(file_path):
        print(f"錯誤：找不到音訊檔案 '{file_path}'。請確保檔案存在於正確的路徑。")
        return

    total_time = 0
    successful_requests = 0
    request_durations = [] # 儲存每次成功請求的時間

    print(f"準備對 '{url}' 進行 {count} 次 API 呼叫，使用檔案 '{file_path}'...\n")

    for i in range(count):
        print(f"--- 請求 {i+1}/{count} ---")
        request_start_time = time.time() # 記錄單次請求開始時間
        try:
            with open(file_path, 'rb') as f:
                files = {'audio_file': (os.path.basename(file_path), f)}
                
                response = requests.post(url, files=files)
                
                # 檢查回應狀態碼
                response.raise_for_status() # 如果狀態碼是 4xx 或 5xx，則會引發 HTTPError
                
                request_end_time = time.time() # 記錄單次請求結束時間
                duration = request_end_time - request_start_time
                request_durations.append(duration)
                total_time += duration
                
                print(f"請求成功！耗時：{duration:.4f} 秒")
                print("伺服器回應 (JSON):")
                try:
                    print(response.json())
                    successful_requests += 1
                except requests.exceptions.JSONDecodeError:
                    print("錯誤：伺服器回應不是有效的 JSON 格式。")
                    print("原始回應內容：")
                    print(response.text)
                

        except requests.exceptions.RequestException as e:
            request_end_time = time.time() # 記錄單次請求結束時間 (即使失敗)
            duration = request_end_time - request_start_time
            total_time += duration # 即使失敗也計入總時間
            print(f"請求失敗：{e}")
            print(f"失敗請求耗時：{duration:.4f} 秒")
        except Exception as e:
            request_end_time = time.time() # 記錄單次請求結束時間 (即使出錯)
            duration = request_end_time - request_start_time
            total_time += duration
            print(f"發生未預期的錯誤：{e}")
            print(f"錯誤請求耗時：{duration:.4f} 秒")
        finally:
            print("-" * 20)
        
        # 每次請求之間可以稍微停頓一下，避免對伺服器造成太大壓力（可選）
        # time.sleep(0.1) 

    print("\n--- 測試摘要 ---")
    print(f"總請求次數：{count}")
    print(f"成功請求次數：{successful_requests}")
    print(f"失敗請求次數：{count - successful_requests}")
    print(f"所有請求總耗時 (包含開啟檔案、網路傳輸、伺服器處理)：{total_time:.4f} 秒")
    
    if successful_requests > 0:
        average_successful_request_time = sum(request_durations) / successful_requests
        print(f"平均成功請求耗時：{average_successful_request_time:.4f} 秒")
        if len(request_durations) > 1: # 避免只有一次成功請求時計算標準差出錯
            min_time = min(request_durations)
            max_time = max(request_durations)
            print(f"最短成功請求耗時：{min_time:.4f} 秒")
            print(f"最長成功請求耗時：{max_time:.4f} 秒")

    if count > 0:
        average_overall_request_time = total_time / count
        print(f"平均每次請求耗時 (包含成功與失敗)：{average_overall_request_time:.4f} 秒")


if __name__ == "__main__":
    # 執行測試前，請確保 server.py 正在運行，並且 audio.mp3 檔案存在。
    print(f"正在嘗試使用 '{audio_file_path}' 測試 API: {url}")
    test_captcha_api_multiple_times(audio_file_path, num_requests)