FROM python:3.10-bookworm

WORKDIR /app

COPY req_dock.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=200 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r req_dock.txt \
    transformers

COPY . .

CMD ["python3", "app.py"]