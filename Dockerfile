FROM python:3.11-slim

# Minimal system deps for opencv-headless + easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install everything, then FORCE replace opencv-python (GUI) with headless
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir --force-reinstall opencv-python-headless>=4.9.0.80

COPY . .

EXPOSE 8000
CMD ["python3", "server.py"]
