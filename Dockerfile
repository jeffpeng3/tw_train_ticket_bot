# 1. 使用 python:3.8-slim 作為基礎映像
FROM python:3.8-slim

# 2. 設定工作目錄
WORKDIR /app

# 3. 安裝系統依賴項和 uv
# 更新 apt 套件列表並安裝依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # 安裝 uv
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# 設定 uv 的路徑
ENV PATH="/root/.local/bin:$PATH"

# 4. 複製 requirements.txt
COPY requirements.txt .

# 5. 使用 uv 安裝 Python 依賴項
# --system 選項讓 uv 使用系統 Python 環境，類似於傳統的 pip install
# --no-cache 避免快取問題
RUN uv pip install --system --no-cache -r requirements.txt

# 6. 複製專案中的其餘所有檔案
COPY . .

# 建立 log 目錄
RUN mkdir -p log

# 7. 公開端口 (預設 8000)
EXPOSE 8000

# 8. 設定 CMD 以執行 gunicorn
# 使用 exec 格式以確保 gunicorn 作為 PID 1 運行
CMD ["gunicorn", "-b", "0.0.0.0:8000", "server:app"]