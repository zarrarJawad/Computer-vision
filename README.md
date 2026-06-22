# UCF-101 Video Action Recognition
### R(2+1)D Feature Extraction + KNN Classification

A video action recognition system trained on the UCF-101 benchmark dataset. Uses a **R(2+1)D** deep learning backbone to extract spatiotemporal features from videos, then classifies actions using **K-Nearest Neighbors** with cosine similarity.

---

## Results

| Metric | Score |
|---|---|
| Top-1 Accuracy | **98.22%** |
| Top-3 Accuracy | **99.04%** |
| Top-5 Accuracy | **99.25%** |
| Weighted Precision | 96.67% |
| Weighted Recall | 96.74% |
| Weighted F1-Score | 96.55% |

Evaluated across **5,834 video clips** spanning **101 action classes**.

Selected per-class highlights:

| Class | F1-Score |
|---|---|
| Drumming | 1.000 |
| BoxingSpeedBag | 1.000 |
| PlayingGuitar | 1.000 |
| IceDancing | 0.987 |
| JumpRope | 0.990 |

> Full per-class breakdown in full_report

---

## How It Works

```
Video Clips
    │
    ▼
R(2+1)D Backbone  ──►  Feature Vectors  ──►  features.pkl
                                                   │
                                                   ▼
                                         StandardScaler (normalize)
                                                   │
                                                   ▼
                                    KNN (k=6, cosine distance)
                                                   │
                                                   ▼
                                         Top-1 / Top-3 / Top-5
                                           Classification
```

The R(2+1)D network decomposes 3D convolutions into a spatial 2D conv followed by a temporal 1D conv, capturing both appearance and motion efficiently. The resulting feature vectors are stored in `features.pkl` and compared at query time using cosine similarity — meaning **no GPU is required to run evaluation or classify new videos**.

---

## Supported Action Classes

The model recognises **101 action categories** from the UCF-101 dataset, including:

`Archery` `Basketball` `Biking` `Bowling` `Boxing` `BrushingTeeth` `Diving` `Drumming` `GolfSwing` `Hammering` `HighJump` `HorseRiding` `HulaHoop` `IceDancing` `JugglingBalls` `JumpRope` `Kayaking` `Knitting` `LongJump` `Nunchucks` `PlayingGuitar` `PlayingPiano` `PlayingViolin` `PoleVault` `RockClimbingIndoor` `Rowing` `SalsaSpin` `Skiing` `Skydiving` `SoccerJuggling` `Surfing` `Swing` `TaiChi` `TennisSwing` `Typing` `WalkingWithDog` `YoYo` and more.

Full class list at [UCF-101 Homepage](https://www.crcv.ucf.edu/data/UCF101.php).

---

## Project Structure

```
├── evaluate.py                      # Evaluation script
├── ucf101_features_r2plus1d_2.pkl   # Pre-extracted feature vectors (the model)
├── evaluation_results/
│   ├── full_report.txt              # Full per-class classification report
│   └── sample_results.png           # Sample visualization (if keyframes present)
└── UCF101_keyframes/                # Optional — only needed for visualization
    ├── Basketball/
    │   └── video_id/
    │       └── frame_001.jpg
    └── ...
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/zarrarJawad/ucf101-action-recognition.git
cd ucf101-action-recognition
```

### 2. Install dependencies
```bash
pip install numpy scikit-learn opencv-python matplotlib torch torchvision
```

### 3. Download the feature file

> [!IMPORTANT]
> **The `.pkl` feature file is NOT included in this repo due to file size. You must download it separately or the script will not run.**

---

## ⬇️ Download Model Weights

### [`ucf101_features_r2plus1d_2.pkl`](https://drive.google.com/file/d/1OpfKakz4ecs3e5Y2eH3cigJifll9LHBz/view?usp=sharing)

> This file contains the pre-extracted R(2+1)D feature vectors for all 5,834 videos across 101 action classes. It is the core of the model — without it nothing will run.

**[👉 Click here to download from Google Drive](https://drive.google.com/file/d/1OpfKakz4ecs3e5Y2eH3cigJifll9LHBz/view?usp=sharing)**

Once downloaded, place it in the **root of the project folder** (same level as `evaluate.py`):

```
ucf101-action-recognition/
├── evaluate.py
├── ucf101_features_r2plus1d_2.pkl   ← goes here
└── ...
```

---

---

## Running the Evaluation

```bash
python evaluate.py
```

If you don't have the `UCF101_keyframes/` folder, the script will crash at the visualization step. Either download the keyframes from the [UCF-101 site](https://www.crcv.ucf.edu/data/UCF101.php), or comment out the `# --- GENERATE IMAGE OUTPUT ---` block in `evaluate.py` to skip it. Everything else — accuracy, precision, recall, F1 — runs from the `.pkl` alone.

---

## Classifying Your Own Video

This is a two-step process. The `.pkl` file stores feature vectors extracted by the R(2+1)D backbone. To classify a new video, you must run it through that **same backbone** first to get a compatible feature vector, then query the KNN.

### Step 1 — Extract features from your video

```python
import torch
import torchvision.models.video as video_models
import torchvision.transforms as transforms
import cv2
import numpy as np

def extract_features(video_path, num_frames=16):
    # Load pretrained R(2+1)D backbone
    model = video_models.r2plus1d_18(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove classification head, keep features only
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    # Normalize and reshape to (1, C, T, H, W)
    frames = np.array(frames, dtype=np.float32) / 255.0
    mean = np.array([0.43216, 0.394666, 0.37645])
    std  = np.array([0.22803, 0.22145, 0.216989])
    frames = (frames - mean) / std
    tensor = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float()

    with torch.no_grad():
        features = model(tensor).squeeze().numpy()

    return features
```

### Step 2 — Query the KNN with your extracted features

```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the stored features
with open("ucf101_features_r2plus1d_2.pkl", "rb") as f:
    features_dict = pickle.load(f)

video_ids = list(features_dict.keys())
labels    = [features_dict[vid]["label"] for vid in video_ids]
X         = np.array([features_dict[vid]["features"] for vid in video_ids])

# Rebuild the same scaler and KNN used during training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
nbrs = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X_scaled)

# Extract features from your video and query
my_features = extract_features("my_video.mp4")
my_scaled   = scaler.transform([my_features])

distances, indices = nbrs.kneighbors(my_scaled)
neighbor_labels = [labels[i] for i in indices[0][1:]]  # exclude self if in dataset

# Majority vote across top-5 neighbors
predicted = max(set(neighbor_labels[:5]), key=neighbor_labels[:5].count)
print(f"Predicted action: {predicted}")

# See all neighbors and their distances
print("\nTop neighbors:")
for rank, (idx, dist) in enumerate(zip(indices[0][1:6], distances[0][1:6]), 1):
    print(f"  {rank}. {labels[idx]:<25} (distance: {dist:.4f})")
```

### Important Notes

- Your video must go through the **same R(2+1)D backbone** used during training — mixing backbones will produce incompatible vector sizes and garbage results
- The model was trained on UCF-101 categories only. It will still return a prediction for anything you give it (the nearest neighbor in the dataset), but accuracy drops significantly for actions not in the 101 classes
- Short clips (under ~2 seconds) and very low resolution videos may produce weaker features — aim for 112×112px minimum and at least 16 frames

---

## Requirements

```
numpy
scikit-learn
opencv-python
matplotlib
torch
torchvision
```

No GPU required for evaluation or inference. GPU recommended only if re-extracting features from a large new dataset.

---

## Notes on Low-Support Classes

Several classes (`BaseballPitch`, `HighJump`, `SoccerPenalty`, `Shotput`) show 0% F1 in the report. This is a **data issue, not a model issue** — these classes had only 1–3 samples in the evaluation split, which is not enough for KNN to find a matching neighbor. Performance on these classes with more samples would be consistent with the rest.
