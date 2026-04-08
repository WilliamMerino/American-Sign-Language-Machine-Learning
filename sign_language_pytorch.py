"""
=============================================================
  American Sign Language Recognition — Full PyTorch Pipeline
  CNN with Train / Validation / Test Split
=============================================================
  Requirements:
    pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn

  Dataset files expected (same folder as this script):
    sign_mnist_train.csv
    sign_mnist_test.csv
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 30
LR            = 1e-3
VAL_SPLIT     = 0.15      # 15% of train → validation
PATIENCE      = 5         # early stopping patience
OUT_DIR       = "outputs"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# J=9, Z=25 are motion-based and excluded from the dataset
LABEL_MAP = {i: chr(ord('A') + i) for i in range(26) if i not in (9, 25)}

os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Device: {DEVICE}")


# ===========================================================================
# 1.  DATASET
# ===========================================================================
class SignMNISTDataset(Dataset):
    """
    Loads the Sign Language MNIST CSV.
    Each row: label, pixel1..pixel784 (28x28 grayscale, 0-255)
    Returns normalised float tensor [1, 28, 28] and integer label.
    """
    def __init__(self, dataframe, augment=False):
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.long)
        pixels = dataframe.drop("label", axis=1).values.astype(np.float32) / 255.0
        # shape: (N, 784) → (N, 1, 28, 28)
        self.images = torch.tensor(pixels).reshape(-1, 1, 28, 28)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            # Random brightness jitter ±10%
            img = (img + (torch.rand(1).item() - 0.5) * 0.2).clamp(0, 1)

        return img, label


# ===========================================================================
# 2.  MODEL — CNN
# ===========================================================================
class SignCNN(nn.Module):
    """
    Architecture:
      Conv block 1: 1  → 32  filters, 3×3, BN, ReLU, MaxPool
      Conv block 2: 32 → 64  filters, 3×3, BN, ReLU, MaxPool
      Conv block 3: 64 → 128 filters, 3×3, BN, ReLU, MaxPool
      FC: 128*3*3 → 256 → Dropout(0.5) → 24 classes
    """
    def __init__(self, num_classes=24):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28×28 → 14×14
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14×14 → 7×7
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 7×7 → 3×3
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# 3.  LOAD DATA & SPLIT
# ===========================================================================
print("=" * 60)
print("  1 · LOADING DATA & SPLITTING")
print("=" * 60)

train_df = pd.read_csv("sign_mnist_train.csv")
test_df  = pd.read_csv("sign_mnist_test.csv")

# Build full training dataset (with augmentation)
full_train_dataset = SignMNISTDataset(train_df, augment=True)

# Train / Validation split (85% / 15%)
n_total = len(full_train_dataset)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(
    full_train_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

# Validation and test should NOT use augmentation
val_dataset.dataset  = SignMNISTDataset(train_df, augment=False)
test_dataset         = SignMNISTDataset(test_df, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  Training samples   : {n_train:,}")
print(f"  Validation samples : {n_val:,}")
print(f"  Test samples       : {len(test_dataset):,}")
print(f"  Classes            : {len(LABEL_MAP)} ASL letters (excl. J & Z)")


# ===========================================================================
# 4.  EDA
# ===========================================================================
print("\n" + "=" * 60)
print("  2 · EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0f0f1a")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

unique_labels = sorted(train_df["label"].unique())
label_names   = [LABEL_MAP[l] for l in unique_labels]

# Class distribution
ax1 = fig.add_subplot(gs[0, :])
counts = train_df["label"].value_counts().sort_index()
colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(counts)))
bars = ax1.bar(label_names, counts.values, color=colors, edgecolor="none", width=0.7)
ax1.set_facecolor("#0f0f1a")
ax1.set_title("Class Distribution — Full Training CSV", color="white", fontsize=16, fontweight="bold", pad=12)
ax1.set_xlabel("ASL Letter", color="#aaaacc", fontsize=12)
ax1.set_ylabel("Sample Count", color="#aaaacc", fontsize=12)
ax1.tick_params(colors="white")
for spine in ax1.spines.values(): spine.set_edgecolor("#333355")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             str(val), ha="center", va="bottom", color="white", fontsize=8)
ax1.set_ylim(0, counts.max() * 1.12)
ax1.grid(axis="y", color="#222244", linewidth=0.5)

# Split pie
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#0f0f1a")
sizes  = [n_train, n_val, len(test_dataset)]
slabels = [f"Train\n{n_train:,}", f"Validation\n{n_val:,}", f"Test\n{len(test_dataset):,}"]
colors2 = ["#e040fb", "#00e5ff", "#ffd93d"]
wedges, texts, autotexts = ax2.pie(
    sizes, labels=slabels, colors=colors2, autopct="%1.1f%%",
    pctdistance=0.6, textprops={"color": "white", "fontsize": 11},
    wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2}
)
for at in autotexts: at.set_fontsize(10)
ax2.set_title("Train / Val / Test Split", color="white", fontsize=13, fontweight="bold")

# Mean images per class
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0f0f1a")
ax3.set_title("Mean Image per Class", color="white", fontsize=13, fontweight="bold")
pixel_cols  = [c for c in train_df.columns if c != "label"]
n_cols      = 6
n_rows      = (len(unique_labels) + n_cols - 1) // n_cols
composite   = np.zeros((28 * n_rows, 28 * n_cols))
for idx, lbl in enumerate(unique_labels):
    mean_img = train_df[train_df["label"] == lbl][pixel_cols].values.mean(axis=0).reshape(28, 28)
    r, c = divmod(idx, n_cols)
    composite[r*28:(r+1)*28, c*28:(c+1)*28] = mean_img
ax3.imshow(composite, cmap="magma", interpolation="nearest")
ax3.axis("off")
for idx, lbl in enumerate(unique_labels):
    r, c = divmod(idx, n_cols)
    ax3.text(c*28 + 14, r*28 + 26, LABEL_MAP[lbl],
             ha="center", va="bottom", color="white", fontsize=7, fontweight="bold")

# Pixel intensity histogram
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor("#0f0f1a")
sample_pix = train_df[pixel_cols].values[:2000].flatten()
ax4.hist(sample_pix, bins=60, color="#7b5ea7", edgecolor="none", alpha=0.85)
ax4.set_title("Pixel Intensity Distribution", color="white", fontsize=13, fontweight="bold")
ax4.set_xlabel("Pixel Value (0–255)", color="#aaaacc")
ax4.set_ylabel("Frequency", color="#aaaacc")
ax4.tick_params(colors="white")
for spine in ax4.spines.values(): spine.set_edgecolor("#333355")
ax4.grid(color="#222244", linewidth=0.5)

# Sample grid
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#0f0f1a")
ax5.set_title("Random Sample Images", color="white", fontsize=13, fontweight="bold")
show_labels = unique_labels[:4]
sample_grid = np.zeros((28 * 4, 28 * 6))
for ri, lbl in enumerate(show_labels):
    rows = train_df[train_df["label"] == lbl][pixel_cols].values
    chosen = rows[np.random.choice(len(rows), 6, replace=False)]
    for ci, row in enumerate(chosen):
        sample_grid[ri*28:(ri+1)*28, ci*28:(ci+1)*28] = row.reshape(28, 28)
ax5.imshow(sample_grid, cmap="gray", interpolation="nearest")
ax5.axis("off")
for ri, lbl in enumerate(show_labels):
    ax5.text(-3, ri*28 + 14, LABEL_MAP[lbl], ha="right", va="center",
             color="#e040fb", fontsize=12, fontweight="bold")

fig.suptitle("ASL Sign Language MNIST — Exploratory Data Analysis",
             color="white", fontsize=20, fontweight="bold", y=0.98)
plt.savefig(f"{OUT_DIR}/1_eda.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ EDA saved → 1_eda.png")


# ===========================================================================
# 5.  TRAINING
# ===========================================================================
print("\n" + "=" * 60)
print("  3 · TRAINING — CNN with PyTorch")
print("=" * 60)

num_classes = len(LABEL_MAP)
model       = SignCNN(num_classes=num_classes).to(DEVICE)
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Max epochs      : {EPOCHS}   |   Early stopping patience: {PATIENCE}")
print(f"  Optimiser       : Adam (LR={LR}, weight_decay=1e-4)")
print(f"  Scheduler       : ReduceLROnPlateau\n")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc  = 0.0
patience_ctr  = 0
best_weights  = None


def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds      = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    scheduler.step(va_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)

    print(f"  Epoch {epoch:02d}/{EPOCHS}  "
          f"Train loss={tr_loss:.4f} acc={tr_acc*100:.2f}%  |  "
          f"Val loss={va_loss:.4f} acc={va_acc*100:.2f}%")

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
        print(f"             ✓ New best val acc: {best_val_acc*100:.2f}%  (model saved)")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# Restore best model
model.load_state_dict(best_weights)
torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")
print(f"\n  ✓ Best model saved → {OUT_DIR}/best_model.pth")


# ===========================================================================
# 6.  EVALUATION ON TEST SET
# ===========================================================================
print("\n" + "=" * 60)
print("  4 · EVALUATION ON TEST SET")
print("=" * 60)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = accuracy_score(all_labels, all_preds)
label_names_test = [LABEL_MAP[l] for l in sorted(np.unique(all_labels))]

print(f"\n  Test Accuracy: {test_acc * 100:.2f}%")
print(f"  Best Val Acc : {best_val_acc * 100:.2f}%\n")
print(classification_report(all_labels, all_preds, target_names=label_names_test))


# ===========================================================================
# 7.  PLOTS
# ===========================================================================

# ── Learning curves ──────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(18, 6))
fig2.patch.set_facecolor("#0f0f1a")

for ax, metric, title, colors in zip(
    axes,
    [("train_loss", "val_loss"), ("train_acc", "val_acc")],
    ["Loss Curve", "Accuracy Curve"],
    [("#e040fb", "#00e5ff"), ("#e040fb", "#00e5ff")]
):
    ax.set_facecolor("#0f0f1a")
    ep = range(1, len(history[metric[0]]) + 1)
    ax.plot(ep, history[metric[0]], color=colors[0], linewidth=2, label="Train")
    ax.plot(ep, history[metric[1]], color=colors[1], linewidth=2, linestyle="--", label="Validation")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", color="#aaaacc")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333355")
    ax.grid(color="#222244", linewidth=0.5)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")

axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

fig2.suptitle("Training History", color="white", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_learning_curves.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ Learning curves saved → 2_learning_curves.png")

# ── Confusion matrix ─────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(14, 12))
fig3.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")
cm      = confusion_matrix(all_labels, all_preds, labels=sorted(np.unique(all_labels)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="magma",
            xticklabels=label_names_test, yticklabels=label_names_test,
            ax=ax, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
ax.set_title(f"Normalised Confusion Matrix   (Test Acc = {test_acc*100:.1f}%)",
             color="white", fontsize=15, fontweight="bold")
ax.set_xlabel("Predicted", color="#aaaacc", fontsize=12)
ax.set_ylabel("True",      color="#aaaacc", fontsize=12)
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ Confusion matrix saved → 3_confusion_matrix.png")

# ── Per-class metrics bar chart ───────────────────────────────────────────────
report_dict = classification_report(all_labels, all_preds,
                                    target_names=label_names_test, output_dict=True)
letters   = label_names_test
precision = [report_dict[l]["precision"] for l in letters]
recall    = [report_dict[l]["recall"]    for l in letters]
f1        = [report_dict[l]["f1-score"]  for l in letters]
x, w      = np.arange(len(letters)), 0.27

fig4, ax = plt.subplots(figsize=(20, 6))
fig4.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")
ax.bar(x - w, precision, w, label="Precision", color="#e040fb", alpha=0.85)
ax.bar(x,     recall,    w, label="Recall",    color="#00e5ff", alpha=0.85)
ax.bar(x + w, f1,        w, label="F1-Score",  color="#ffd93d", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(letters, color="white", fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title("Per-Class Precision, Recall & F1-Score", color="white", fontsize=15, fontweight="bold")
ax.set_ylabel("Score", color="#aaaacc")
ax.tick_params(colors="white")
ax.axhline(test_acc, color="#ff6b6b", linewidth=1.5, linestyle="--",
           label=f"Overall Accuracy ({test_acc*100:.1f}%)")
for spine in ax.spines.values(): spine.set_edgecolor("#333355")
ax.grid(axis="y", color="#222244", linewidth=0.5)
ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_per_class_metrics.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ Per-class metrics saved → 4_per_class_metrics.png")

# ── Prediction gallery ────────────────────────────────────────────────────────
test_imgs_all  = test_dataset.images.numpy()
test_lbls_all  = test_dataset.labels.numpy()

sample_idxs = np.random.choice(len(test_dataset), 32, replace=False)
fig5, axes  = plt.subplots(4, 8, figsize=(20, 11))
fig5.patch.set_facecolor("#0f0f1a")

for ax, i in zip(axes.flat, sample_idxs):
    img    = test_imgs_all[i, 0]
    true_l = LABEL_MAP[test_lbls_all[i]]
    pred_l = LABEL_MAP[all_preds[i]]
    color  = "#39ff14" if true_l == pred_l else "#ff4444"
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(f"T:{true_l}  P:{pred_l}", color=color, fontsize=9, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

fig5.suptitle("Prediction Gallery   🟢 Correct   🔴 Wrong",
              color="white", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_prediction_gallery.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("  ✓ Prediction gallery saved → 5_prediction_gallery.png")


# ===========================================================================
# 8.  SUMMARY
# ===========================================================================
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Dataset          : Sign Language MNIST (28×28 grayscale)")
print(f"  Train / Val / Test: {n_train:,} / {n_val:,} / {len(test_dataset):,}")
print(f"  Classes          : {num_classes} ASL letters (excl. J & Z)")
print(f"  Model            : CNN  (Conv×6 + FC×2)")
print(f"  Parameters       : {sum(p.numel() for p in model.parameters()):,}")
print(f"  Device           : {DEVICE}")
print(f"  Best Val Acc     : {best_val_acc * 100:.2f}%")
print(f"  Test Accuracy    : {test_acc * 100:.2f}%")
print(f"\n  All outputs → ./{OUT_DIR}/")
print("=" * 60)