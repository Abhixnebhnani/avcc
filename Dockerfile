FROM python:3.11-slim

# Install system deps for opencv headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install headless opencv FIRST, then everything else
# This prevents ultralytics/easyocr from pulling in the GUI version
RUN pip install --no-cache-dir opencv-python-headless>=4.9.0.80 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "server.py"]
