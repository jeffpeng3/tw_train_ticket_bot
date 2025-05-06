# tw_train_ticket_bot 專案記憶

本文檔總結了 `tw_train_ticket_bot` 專案（特別是 `server.py` 和 `captcha_handler.py`）的主要修改、討論重點以及尚未實施的重構建議。

## 一、已完成的主要修改和優化

*   **模組化**：驗證碼邏輯從 `server.py` 移至 `captcha_handler.py`。
*   **介面封裝**：`captcha_handler.py` 提供 `resolve_audio_captcha` 介面。
*   **結構優化**：`server.py` 使用 `BookingConfig` dataclass，簡化方法參數，推廣 f-string。
*   **資源管理**：
    *   HTTP 請求使用 `requests.Session()`。
    *   日誌輸出到 `stdout`。
    *   臨時檔案使用 `tempfile` 自動清理。
    *   移除 `fileUUID`。
*   **錯誤處理 (captcha_handler.py)**：TFLite 模型載入失敗時快速失敗 (RuntimeError)。
*   **簡化 (captcha_handler.py)**：移除函式內對 TFLite interpreter 的冗餘 `None` 檢查。
*   **效能優化 (captcha_handler.py - I/O)**：音訊片段處理改用記憶體流 (`io.BytesIO`)。
*   **效能優化 (captcha_handler.py - 提早退出)**：任一音訊片段預測為 '?' 或失敗時立即返回。

## 二、關於 `captcha_handler.py` 中 TFLite `interpreter` 的討論總結

*   **目前使用方式**：模組級單一共享實例，循序用於各音訊片段預測。
*   **並行化考量**：
    *   `Interpreter` 本身非線程安全。
    *   推薦策略：每個並行工作單元（線程/進程）創建獨立的 `Interpreter` 實例。
    *   記憶體管理：可通過 `concurrent.futures` 執行器的 `max_workers` 參數控制並行度，以管理記憶體消耗。
*   **模型資訊**：模型檔案 (`model.tflite`) 約 2.4MB，循序處理6個字元約需 0.4 秒（不計模型載入）。

## 三、尚未實施的重構建議 (針對 `captcha_handler.py`)

*   **效能**：
    *   並行處理音訊片段的預測（例如，使用 `concurrent.futures.ProcessPoolExecutor`，為每個工作進程創建獨立 `Interpreter`，並設定合理的 `max_workers`）。
*   **可讀性/維護性**：
    *   改進錯誤處理和日誌記錄機制（使用 `logging` 模組替換 `print`）。
    *   將魔術數字替換為具名常數。