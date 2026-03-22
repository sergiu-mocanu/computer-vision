# Webcam Computer Vision Prototype

Real-time computer vision prototype using a webcam to explore:

- Visual anomaly detection (DINOv2)
- Zero-shot image–text similarity (CLIP)

---

## Overview

The application supports two independent modes:

## 1. Anomaly Detection (DINOv2)

Learns a “normal” visual state from reference frames and detects deviations.

Pipeline:

```angular2html
Webcam frame
    ↓
DINOv2 embedding (ViT)
    ↓
Reference comparison (cosine distance)
    ↓
Statistical normalization (z-score using mean/std of normal data)
    ↓
Temporal smoothing (EMA)
    ↓
Anomaly score
```

## 2. Semantic Labeling (CLIP)

Scores how well the current frame matches predefined text prompts (text-image similarity).

Example prompts:

- a hand in front of the camera
- a person in front of the camera
- an empty chair
- a mirror

Output:
- Best matching prompt
- Confidence score
- Top-k ranked prompts


## 3. Pipeline Mode (Anomaly → Semantic Labeling)

Combine `anomaly detection` + `image labeling` to reduce computational cost:

- Record, compute and label a normal scene
- Use DINOv2 to detect anomalies
- Use CLIP's text-image similarity only when an anomaly is detected

```angular2html
Webcam frame
    ↓
DINOv2 embedding (ViT)
    ↓
Reference comparison (cosine distance)
    ↓
Temporal smoothing (EMA)
    ↓
Anomaly score
    ↓
[if anomaly detected]
    ↓
CLIP (image–text similarity)
    ↓
Semantic label + confidence
```

---

## Features
- Real-time webcam inference (OpenCV)
- Vision Transformer embeddings (DINOv2)
- Zero-shot multimodal reasoning (CLIP)
- Configurable runtime behavior
- GPU acceleration (PyTorch + CUDA)

---

## Project structure

```angular2html
src/webcam_cv/
├── app.py
├── config.py
├── camera.py
├── display.py
├── models/
│   ├── base.py
│   ├── dinov2_embedder.py
│   ├── clip_embedder.py
│   ├── factory.py
│   └── registry.py
├── app_modes/
│   ├── anomaly_app.py
│   ├── labeling_app.py.py
│   └── pipeline_app.py
├── anomaly/
│   └── scorer.py
├── experiments
│   └── resolution_benchmark.py
└── utils/
    └── image.py
```

---

## Installation

Recommended (conda):

```bash
conda env create -f environment.yml
conda activate computer-vision
```

Alternative (pip):
```bash
pip install -r requirements.txt
```

The two are intended to provide similar runtime capability, but the conda environment is the reference setup for native 
CV dependencies.

### GPU support (optional)

For GPU acceleration, install PyTorch with CUDA support using the official selector:

https://pytorch.org/get-started/locally/

Select:
- Linux
- pip or conda
- CUDA version compatible with your system

If CUDA is not available, the prototype will run on CPU (slower but functional).

---

## Running
```bash
python main.py
```

---

## Configuration

Edit `config.py`

### Select app mode

```python
model_type = 'anomaly'  # or 'labeling', 'pipeline'
model_size = 'base'  # or 'None'
```

---

## Controls

### Anomaly mode | Pipeline mode

| Key | Action                  |
| --- | ----------------------- |
| r   | record reference frames |
| c   | clear reference         |
| s   | save frame              |
| q   | quit                    |


### Labeling mode

| Key | Action     |
| --- | ---------- |
| s   | save frame |
| q   | quit       |


---

## Available Models

| Model  | Variants           | Purpose                     |
| ------ | ------------------ | --------------------------- |
| dinov2 | small, base, large | visual embeddings / anomaly |
| clip   | base, large        | image–text similarity       |

---

## Current limitations

This is an early prototype.

Limitations include:
- Frame-level anomaly detection (no localization)
- Fixed prompt list for labeling
- Threshold tuning is manual

---

## License

This project is licensed under the terms of the MIT license.