# Jaundice Detection (with Baby Presence Gate) – Architecture & Operations

## What it Does
Real‑time Streamlit app that:
- Confirms a baby/face is present (presence gate).
- Screens for neonatal jaundice (Normal vs Jaundice) with a MobileNetV3 classifier.
- Blocks/flags low‑quality frames (dark / no baby) to cut false positives.

## Key Components
- **Streamlit UI** (`app.py`): upload, webcam snapshot, or live video feed; overlays predictions/brightness/reliability on frames.
- **Jaundice classifier**: MobileNetV3‑Small binary head (sigmoid) with weights `jaundice_mobilenetv3_robust.pt`.
- **Presence gate** (`baby_presence_detector.py`): Haar face detection + YCrCb skin‑ratio heuristic to confirm a baby/face before running the classifier.
- **Brightness gate**: rejects too‑dark frames; lowers reliability in low light.

## Data Flow (Live Mode)
```
Camera frame
   ↓
[Brightness check]
   ├─ Too dark → “Too Dark” (stop)
   └─ Pass
   ↓
[Presence gate: face + skin mask]
   ├─ No face/skin → “No Baby Detected” (stop)
   └─ Pass
   ↓
[Preprocess: resize 224, normalize]
   ↓
[MobileNetV3 sigmoid]
   ↓
Label + prob + reliability overlay → Streamlit display
```

## Baby Detection Technique (Presence Gate)
- **Face detector**: OpenCV Haar cascade (`haarcascade_frontalface_default.xml`).
- **Skin exposure check**: YCrCb skin mask; require min skin ratio.
- **Heuristics**: min face area ratio to avoid tiny/false faces; both face + skin must pass.
- **Output**: `is_present`, `face_count`, `max_face_area_ratio`, `skin_ratio`, `reason` (why it failed).

## Jaundice Model
- **Backbone/head**: `torchvision.mobilenet_v3_small` with a binary sigmoid head.
- **Weights**: `jaundice_mobilenetv3_robust.pt`.
- **Input**: 224×224 RGB, ImageNet mean/std.
- **Decision**: probability > 0.5 ⇒ Jaundice else Normal.

## Interfaces
- **Run UI**: `streamlit run jaundice_detection/app.py`
- **Presence CLI test**: `python jaundice_detection/baby_presence_detector.py path/to/image.jpg --save-mask mask.png`
  - Prints: `is_present`, `face_count`, `max_face_area_ratio`, `skin_ratio`, `reason`

## Configuration Knobs
- `app.py`
  - `IMG_SIZE=224`, `MEAN/STD` (ImageNet), decision threshold `0.5` on sigmoid.
  - `brightness_threshold` (default 35; frames < threshold → “Too Dark”; low light → reliability down‑weight).
- `baby_presence_detector.py`
  - `min_face_area_ratio` (default 0.015): minimum face area relative to frame.
  - `min_skin_ratio` (default 0.01): minimum exposed‑skin ratio.
  - Haar params: `scaleFactor=1.1`, `minNeighbors=4`.

## Reliability Measures
- **Brightness gating**: blocks very dark frames; annotates low‑light reliability.
- **Presence gating**: requires both a face and skin exposure to proceed.
- **Overlays**: class + probability + brightness + reliability (low‑light note).

## Files & Responsibilities
- `jaundice_detection/app.py`: UI, model load, gating, overlay logic, live/shot/upload handling.
- `jaundice_detection/baby_presence_detector.py`: face/skin gate and CLI; YCrCb skin mask + Haar face detector.
- `jaundice_detection/jaundice_mobilenetv3_robust.pt`: model weights (binary jaundice classifier).

## Pipeline Architecture (Concise)
- Ingestion: camera → frame buffer (Streamlit live feed).
- Gate 1: brightness check (block too-dark; mark low-light).
- Gate 2: presence check (Haar face + skin ratio).
- Preprocess: resize/crop/normalize to 224.
- Inference: MobileNetV3 binary sigmoid.
- Postprocess: class/prob + overlays (brightness, reliability, presence reason).
- UI: Streamlit image/video rendering; upload/snapshot/live controls.

## Suggested Improvements
- **Baby detection**
  - Swap Haar for a lightweight detector (e.g., MobileNet-SSD/YOLOv5n/YOLOv8n) to improve recall/robustness.
  - Add temporal smoothing (require presence over N consecutive frames) to avoid flicker.
  - Add crib ROI masking if camera is fixed; ignore regions outside crib.
  - Use face landmarks or HOG+SVM as a fallback when Haar fails.
  - Track brightness+skin stats over time to auto-tune thresholds.
- **Jaundice detection**
  - Calibrate decision threshold per dataset (optimize ROC/PR).
  - Add low-light color correction (gray-world/CLAHE) before inference in low-light cases.
  - Ensemble with a lighter secondary model for cross-check, or add uncertainty estimation (MC-dropout / temperature scaling).
  - Collect hard negatives (empty crib, blankets, bright backgrounds) and fine-tune the classifier to reduce false positives.

## Operational Checklist
- Ensure `jaundice_mobilenetv3_robust.pt` is present.
- Camera accessible (for live or webcam modes).
- Light level adequate; aim to keep brightness > threshold to avoid reliability downgrades.
