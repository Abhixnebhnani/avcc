# AVCC Gantry System — Real YOLOv8 Detection

Automatic Vehicle Counter & Classifier using YOLOv8 + FastAPI + WebSockets.
Supports Indian traffic classes: Car, Motorcycle, Auto (rickshaw), Bus,
Truck/HGV, LCV/Van, Tractor, Bicycle.

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> First run will auto-download YOLOv8 nano weights (~6 MB).
> If you have a GPU, install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### 2. Start the server

```bash
python server.py
```

You should see:
```
==================================================
  AVCC Gantry System — Server
==================================================
  YOLOv8: ✅ Ready
  Open: http://localhost:8000
==================================================
```

### 3. Open the dashboard

Open your browser and go to: **http://localhost:8000**

Or open `frontend/index.html` directly in your browser.

### 4. Connect & run

1. Click **Connect** (top right of dashboard)
2. Drop or upload your `.mp4` / `.avi` video file
3. Adjust the **counting line** position using the slider
4. Click **▶ Play**
5. Watch detections, counts, and classifications in real time

---

## How it works

```
Video frame (from browser)
    ↓  WebSocket
FastAPI server receives JPEG frame
    ↓
YOLOv8 nano runs inference
    ↓
Bounding boxes filtered to vehicles only
    ↓
Indian class refinement:
  - small "car" → Motorcycle / Auto
  - boxy car → Auto (3-wheeler)
  - compact truck → LCV/Van or Tractor
    ↓
Centroid tracker assigns vehicle IDs
    ↓
Counting line crossing detected
    ↓
Speed calculated (pixel displacement × scale factor)
    ↓
Record: [id | time | lane | class | speed | confidence]
    ↓
Annotated frame sent back to browser
    ↓
Dashboard updates live
```

---

## Counting Line

Adjust the slider in the dashboard to move the red counting line.
Place it somewhere all vehicles must cross — ideally in the middle
of the frame where they are fully visible.

---

## Speed Accuracy Note

Speed is estimated from pixel displacement between frames.
For accurate results, you need to calibrate the scale factor:

In `server.py`, find this line in `update_trackers()`:
```python
scale = 4.0 / (0.12 * state.frame_h)   # metres per pixel
```

Change `4.0` to the real-world length (in metres) of a reference vehicle
visible in your footage, and `0.12` to the fraction of frame height it occupies.

For a proper gantry setup with known camera height and angle,
replace with a homography calibration matrix.

---

## Performance

| Hardware        | Approx FPS |
|-----------------|-----------|
| CPU only        | 3–8 fps   |
| GPU (GTX 1060+) | 20–30 fps |
| GPU (RTX 3060+) | 30–60 fps |

The system sends frames at 10 fps to the server by default.
Adjust `FRAME_INTERVAL_MS` in `frontend/index.html` to change this.

---

## Files

```
avcc_system/
├── server.py          ← FastAPI + YOLOv8 backend
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── frontend/
    └── index.html     ← Browser dashboard
```

---

## Troubleshooting

**"YOLOv8 not available"**
→ Run: `pip install ultralytics`

**WebSocket won't connect**
→ Make sure `python server.py` is running and shows port 8000

**No detections**
→ Try lowering confidence threshold in server.py: change `conf=0.35` to `conf=0.25`

**Very slow on CPU**
→ Increase `FRAME_INTERVAL_MS` in frontend/index.html from 100 to 300
→ Or upgrade to YOLOv8n (already default, the smallest model)

**Motorcycle/Auto misclassified**
→ The class refinement heuristics in `refine_class()` in server.py can be tuned
   to your specific camera angle and traffic mix
