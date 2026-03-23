FROM python:3.11-slim

# Minimal deps for opencv-headless (no GL needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless>=4.9.0.80 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "server.py"]
