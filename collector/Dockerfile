FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import redis; r = redis.Redis('redis', 6379); r.ping()" || exit 1

CMD ["python", "collector.py"]
