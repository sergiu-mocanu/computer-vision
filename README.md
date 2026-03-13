# Webcam Visual Anomaly Detection (Prototype)

Small experimental prototype for real-time visual anomaly detection using a webcam.

The goal of this project is to explore computer vision foundations such as vision transformers, feature embeddings, and similarity-based anomaly detection.

This repository was created as preparation for roles involving industrial computer vision and applied AI.

---

## Overview

The system learns a "normal" visual state from a short recording of webcam frames and then detects when the scene deviates from that reference.

Pipeline:

Webcam frame  
→ Vision transformer feature extraction (DINOv2)  
→ Image embedding  
→ Similarity comparison with reference embedding  
→ Anomaly score

---

## Demo

Example workflow:

1. Run the script
2. Press **r** to record normal frames
3. Modify the scene (add/remove an object)
4. The anomaly score increases

---

## Technologies

- Python
- PyTorch
- OpenCV
- Vision Transformers (DINOv2)
- HuggingFace Transformers

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

---

## Running the prototype
| Key | Action                  |
|-----|-------------------------|
| r   | record normal reference |
| c   | clear reference         |
| s   | save frame              |
| q   | quit                    |

---

## Current limitations

This is an early prototype.

Limitations include:
- anomaly detection only at frame level
- no localization of anomalies
- sensitivity to lighting and camera motion
- threshold chosen empirically

---

## License

This project is licensed under the terms of the MIT license.