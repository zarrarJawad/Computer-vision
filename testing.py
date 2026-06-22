import os
import numpy as np
import pickle
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- PATHS --- #
FEATURE_PATH = "ucf101_features_r2plus1d_2.pkl"
KEYFRAMES_ROOT = "UCF101_keyframes"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD FEATURES --- #
print("Loading pre-computed features...")
with open(FEATURE_PATH, "rb") as f:
    features_dict = pickle.load(f)

video_ids = list(features_dict.keys())
labels = [features_dict[vid]["label"] for vid in video_ids]
X = np.array([features_dict[vid]["features"] for vid in video_ids])

# --- BUILD KNN --- #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
nbrs = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X_scaled)

# --- EVALUATION --- #
print("\nRunning evaluation...")

# Top-k Accuracy
distances, neighbor_indices = nbrs.kneighbors(X_scaled)
neighbor_indices = neighbor_indices[:, 1:]  # Exclude self

unique_labels = sorted(list(set(labels)))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
y_true = [label_to_id[label] for label in labels]

top_k_results = {}
for k in [1, 3, 5]:
    correct = sum(
        1 for i, true in enumerate(y_true) if true in [label_to_id[labels[idx]] for idx in neighbor_indices[i][:k]])
    top_k_results[f"Top-{k}"] = correct / len(y_true)

# Classification Report
y_pred = [max(set([labels[idx] for idx in neighbor_indices[i][:5]]),
              key=[labels[idx] for idx in neighbor_indices[i][:5]].count)
          for i in range(len(X_scaled))]
report = classification_report(labels, y_pred, output_dict=True)

# --- GENERATE IMAGE OUTPUT --- #
plt.figure(figsize=(12, 6))
sample_videos = random.sample(video_ids, 5)

for i, vid in enumerate(sample_videos):
    cls = features_dict[vid]["label"]
    path = os.path.join(KEYFRAMES_ROOT, cls, vid)
    frame = random.choice([f for f in os.listdir(path) if f.endswith(".jpg")])

    img = cv2.imread(os.path.join(path, frame))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"{cls}\n{vid}", fontsize=9)
    plt.axis('off')

plt.tight_layout()
image_path = os.path.join(OUTPUT_DIR, "sample_results.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.close()

# --- SAVE FULL REPORT --- #
with open(os.path.join(OUTPUT_DIR, "full_report.txt"), "w") as f:
    f.write("FULL PERFORMANCE REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Top-1 Accuracy: {top_k_results['Top-1']:.2%}\n")
    f.write(f"Top-3 Accuracy: {top_k_results['Top-3']:.2%}\n")
    f.write(f"Top-5 Accuracy: {top_k_results['Top-5']:.2%}\n\n")
    f.write(f"Average Precision: {report['weighted avg']['precision']:.2%}\n")
    f.write(f"Average Recall: {report['weighted avg']['recall']:.2%}\n")
    f.write(f"Average F1-score: {report['weighted avg']['f1-score']:.2%}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(classification_report(labels, y_pred, digits=4))

# --- PYTHON CONSOLE OUTPUT --- #
print("\nSUMMARY RESULTS:")
print("-" * 50)
print(f"Top-1 Accuracy: {top_k_results['Top-1']:.2%}")
print(f"Top-3 Accuracy: {top_k_results['Top-3']:.2%}")
print(f"Top-5 Accuracy: {top_k_results['Top-5']:.2%}")
print(f"\nAverage Precision: {report['weighted avg']['precision']:.2%}")
print(f"Average Recall: {report['weighted avg']['recall']:.2%}")
print(f"Average F1-score: {report['weighted avg']['f1-score']:.2%}")

print(f"\nResults saved to:")
print(f"- Sample image: {image_path}")
print(f"- Full report: {os.path.join(OUTPUT_DIR, 'full_report.txt')}")