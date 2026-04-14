import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================================
#  American Sign Language Recognition — Custom CNN Only
#  Clean project script for a single-model run
# =============================================================

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.15
PATIENCE = 5
SCHEDULER_FACTOR = 0.5
DATA_DIR = "data"
OUT_DIR = "outputs_customcnn_only"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# J=9 and Z=25 are motion-based and excluded from this dataset
DISPLAY_LABEL_MAP = {i: chr(ord('A') + i) for i in range(26) if i not in (9, 25)}


# =============================================================
# 1. DATASET
# =============================================================
class SignMNISTDataset(Dataset):
    """
    Sign Language MNIST dataset from CSV.
    Returns:
      image: float tensor of shape [1, 28, 28]
      label: contiguous integer label in [0, num_classes - 1]
    """
    def __init__(self, dataframe, label_to_idx, augment=False):
        self.raw_labels = dataframe["label"].values
        self.labels = torch.tensor([label_to_idx[x] for x in self.raw_labels], dtype=torch.long)

        pixels = dataframe.drop("label", axis=1).values.astype(np.float32) / 255.0
        self.images = torch.tensor(pixels).reshape(-1, 1, 28, 28)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].clone()
        label = self.labels[idx]

        if self.augment:
            # Light augmentation that is safer for sign images than horizontal flips
            img = (img + (torch.rand(1).item() - 0.5) * 0.12).clamp(0, 1)
            img = (img + 0.02 * torch.randn_like(img)).clamp(0, 1)

        return img, label


# =============================================================
# 2. MODEL
# =============================================================
class CustomCNN(nn.Module):
    """Deeper custom CNN for ASL image classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.30),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================
# 3. LOAD DATA
# =============================================================
print("\n" + "=" * 70)
print("1 · LOADING DATA")
print("=" * 70)

train_path = os.path.join(DATA_DIR, "sign_mnist_train.csv")
test_path = os.path.join(DATA_DIR, "sign_mnist_test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

raw_unique_labels = sorted(train_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(raw_unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
class_names = [DISPLAY_LABEL_MAP[lbl] for lbl in raw_unique_labels]
num_classes = len(raw_unique_labels)

print(f"Raw labels found       : {raw_unique_labels}")
print(f"Number of classes      : {num_classes}")
print("Image size             : 28x28 grayscale")
print(f"Train CSV samples      : {len(train_df):,}")
print(f"Test CSV samples       : {len(test_df):,}")

full_train_aug = SignMNISTDataset(train_df, label_to_idx, augment=True)
full_train_noaug = SignMNISTDataset(train_df, label_to_idx, augment=False)
test_dataset = SignMNISTDataset(test_df, label_to_idx, augment=False)

indices = torch.randperm(len(full_train_aug), generator=torch.Generator().manual_seed(SEED)).tolist()
n_val = int(len(indices) * VAL_SPLIT)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_dataset = Subset(full_train_aug, train_indices)
val_dataset = Subset(full_train_noaug, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train split samples    : {len(train_dataset):,}")
print(f"Validation samples     : {len(val_dataset):,}")
print(f"Test samples           : {len(test_dataset):,}")


# =============================================================
# 4. EDA
# =============================================================
print("\n" + "=" * 70)
print("2 · EXPLORATORY DATA ANALYSIS")
print("=" * 70)

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0f0f1a")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Class distribution
ax1 = fig.add_subplot(gs[0, :])
counts = train_df["label"].value_counts().sort_index()
colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(counts)))
bars = ax1.bar(class_names, counts.values, color=colors, edgecolor="none", width=0.7)
ax1.set_facecolor("#0f0f1a")
ax1.set_title("Class Distribution — Full Training CSV", color="white", fontsize=16, fontweight="bold", pad=12)
ax1.set_xlabel("ASL Letter", color="#aaaacc", fontsize=12)
ax1.set_ylabel("Sample Count", color="#aaaacc", fontsize=12)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_edgecolor("#333355")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, str(val), ha="center", va="bottom", color="white", fontsize=8)
ax1.set_ylim(0, counts.max() * 1.12)
ax1.grid(axis="y", color="#222244", linewidth=0.5)

# Split pie
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#0f0f1a")
sizes = [len(train_dataset), len(val_dataset), len(test_dataset)]
slabels = [f"Train\n{len(train_dataset):,}", f"Validation\n{len(val_dataset):,}", f"Test\n{len(test_dataset):,}"]
colors2 = ["#e040fb", "#00e5ff", "#ffd93d"]
wedges, texts, autotexts = ax2.pie(
    sizes, labels=slabels, colors=colors2, autopct="%1.1f%%",
    pctdistance=0.6, textprops={"color": "white", "fontsize": 11},
    wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(10)
ax2.set_title("Train / Val / Test Split", color="white", fontsize=13, fontweight="bold")

# Mean images per class
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0f0f1a")
ax3.set_title("Mean Image per Class", color="white", fontsize=13, fontweight="bold")
pixel_cols = [c for c in train_df.columns if c != "label"]
n_cols = 6
n_rows = (len(raw_unique_labels) + n_cols - 1) // n_cols
composite = np.zeros((28 * n_rows, 28 * n_cols))
for idx, lbl in enumerate(raw_unique_labels):
    mean_img = train_df[train_df["label"] == lbl][pixel_cols].values.mean(axis=0).reshape(28, 28)
    r, c = divmod(idx, n_cols)
    composite[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = mean_img
ax3.imshow(composite, cmap="magma", interpolation="nearest")
ax3.axis("off")
for idx, lbl in enumerate(raw_unique_labels):
    r, c = divmod(idx, n_cols)
    ax3.text(c * 28 + 14, r * 28 + 26, DISPLAY_LABEL_MAP[lbl], ha="center", va="bottom", color="white", fontsize=7, fontweight="bold")

# Pixel histogram
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor("#0f0f1a")
sample_pix = train_df[pixel_cols].values[:2000].flatten()
ax4.hist(sample_pix, bins=60, color="#7b5ea7", edgecolor="none", alpha=0.85)
ax4.set_title("Pixel Intensity Distribution", color="white", fontsize=13, fontweight="bold")
ax4.set_xlabel("Pixel Value (0–255)", color="#aaaacc")
ax4.set_ylabel("Frequency", color="#aaaacc")
ax4.tick_params(colors="white")
for spine in ax4.spines.values():
    spine.set_edgecolor("#333355")
ax4.grid(color="#222244", linewidth=0.5)

# Sample grid
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#0f0f1a")
ax5.set_title("Random Sample Images", color="white", fontsize=13, fontweight="bold")
show_labels = raw_unique_labels[:4]
sample_grid = np.zeros((28 * 4, 28 * 6))
for ri, lbl in enumerate(show_labels):
    rows = train_df[train_df["label"] == lbl][pixel_cols].values
    chosen = rows[np.random.choice(len(rows), 6, replace=False)]
    for ci, row in enumerate(chosen):
        sample_grid[ri * 28:(ri + 1) * 28, ci * 28:(ci + 1) * 28] = row.reshape(28, 28)
ax5.imshow(sample_grid, cmap="gray", interpolation="nearest")
ax5.axis("off")
for ri, lbl in enumerate(show_labels):
    ax5.text(-3, ri * 28 + 14, DISPLAY_LABEL_MAP[lbl], ha="right", va="center", color="#e040fb", fontsize=12, fontweight="bold")

fig.suptitle("ASL Sign Language MNIST — Exploratory Data Analysis", color="white", fontsize=20, fontweight="bold", y=0.98)
plt.savefig(f"{OUT_DIR}/1_eda.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"EDA saved -> {OUT_DIR}/1_eda.png")


# =============================================================
# 5. TRAINING UTILITIES
# =============================================================
def create_optimizer_and_scheduler(model):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=SCHEDULER_FACTOR
    )
    return optimizer, scheduler


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def train_model(model_name, model):
    print("\n" + "=" * 70)
    print(f"3 · TRAINING {model_name.upper()}")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(model)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_ctr = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={tr_loss:.4f}, acc={tr_acc * 100:.2f}% | "
            f"val loss={va_loss:.4f}, acc={va_acc * 100:.2f}% | "
            f"lr={current_lr:.6f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            print(f"  New best validation accuracy: {best_val_acc * 100:.2f}%")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"{OUT_DIR}/{model_name.lower()}_best.pth")

    all_labels, all_preds = evaluate_model(model, test_loader)
    test_acc = accuracy_score(all_labels, all_preds)
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n{model_name} test accuracy: {test_acc * 100:.2f}%")
    print(report_text)

    with open(f"{OUT_DIR}/{model_name.lower()}_classification_report.txt", "w") as f:
        f.write(report_text)

    return {
        "model_name": model_name,
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "report_dict": report_dict,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# =============================================================
# 6. TRAIN MODEL
# =============================================================
results = train_model("CustomCNN", CustomCNN(num_classes=num_classes).to(DEVICE))


# =============================================================
# 7. SAVE PLOTS
# =============================================================
print("\n" + "=" * 70)
print("4 · SAVING PLOTS")
print("=" * 70)

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax in axes:
    ax.set_facecolor("#0f0f1a")
fig.patch.set_facecolor("#0f0f1a")

epochs = range(1, len(results["history"]["train_loss"]) + 1)
axes[0].plot(epochs, results["history"]["train_loss"], linewidth=2, label="Train")
axes[0].plot(epochs, results["history"]["val_loss"], linestyle="--", linewidth=2, label="Validation")
axes[1].plot(epochs, results["history"]["train_acc"], linewidth=2, label="Train")
axes[1].plot(epochs, results["history"]["val_acc"], linestyle="--", linewidth=2, label="Validation")

axes[0].set_title("Loss Curves", color="white", fontsize=14, fontweight="bold")
axes[1].set_title("Accuracy Curves", color="white", fontsize=14, fontweight="bold")
for ax in axes:
    ax.set_xlabel("Epoch", color="#aaaacc")
    ax.tick_params(colors="white")
    ax.grid(color="#222244", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
fig.suptitle("Custom CNN Training History", color="white", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_customcnn_training_curves.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()

# Confusion matrix
fig_cm, ax = plt.subplots(figsize=(14, 12))
fig_cm.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")
cm = confusion_matrix(results["all_labels"], results["all_preds"], labels=list(range(num_classes)))
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".0%",
    cmap="magma",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8},
)
ax.set_title(
    f"CustomCNN — Normalised Confusion Matrix (Test Acc = {results['test_acc'] * 100:.1f}%)",
    color="white",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Predicted", color="#aaaacc", fontsize=12)
ax.set_ylabel("True", color="#aaaacc", fontsize=12)
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_customcnn_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()


# =============================================================
# 8. SAVE SUMMARY
# =============================================================
summary = {
    "Model": results["model_name"],
    "Parameters": results["n_params"],
    "Best Val Accuracy": round(results["best_val_acc"] * 100, 2),
    "Test Accuracy": round(results["test_acc"] * 100, 2),
    "Scheduler Factor": SCHEDULER_FACTOR,
    "Batch Size": BATCH_SIZE,
    "Epochs": EPOCHS,
    "Learning Rate": LR,
    "Patience": PATIENCE,
}

pd.DataFrame([summary]).to_csv(f"{OUT_DIR}/4_summary.csv", index=False)
with open(f"{OUT_DIR}/4_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("5 · FINAL SUMMARY")
print("=" * 70)
print(f"Model            : {summary['Model']}")
print(f"Parameters       : {summary['Parameters']:,}")
print(f"Best Val Accuracy: {summary['Best Val Accuracy']:.2f}%")
print(f"Test Accuracy    : {summary['Test Accuracy']:.2f}%")
print(f"Scheduler Factor : {summary['Scheduler Factor']}")
print(f"Outputs saved in : {OUT_DIR}")
print("Done.")
