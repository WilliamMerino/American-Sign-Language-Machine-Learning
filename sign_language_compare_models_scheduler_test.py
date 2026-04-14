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
#  American Sign Language Recognition — Model Comparison
#  Compare LeNet-5 vs Custom CNN on Sign Language MNIST
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
DATA_DIR = "data"
OUT_DIR = "outputs_compare"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# J=9, Z=25 are motion-based and excluded from the dataset
DISPLAY_LABEL_MAP = {i: chr(ord('A') + i) for i in range(26) if i not in (9, 25)}


# =============================================================
# 1. DATASET
# =============================================================
class SignMNISTDataset(Dataset):
    """
    Sign Language MNIST dataset from CSV.
    Returns:
      image: float tensor of shape [1, 28, 28]
      label: contiguous integer label in [0, num_classes-1]
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
            # Safer augmentations for sign-language images:
            # small brightness shift + light Gaussian noise
            img = (img + (torch.rand(1).item() - 0.5) * 0.12).clamp(0, 1)
            img = (img + 0.02 * torch.randn_like(img)).clamp(0, 1)

        return img, label


# =============================================================
# 2. MODELS
# =============================================================
class LeNet5(nn.Module):
    """LeNet-5 style CNN adapted for 28x28 grayscale input."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),      # 28 -> 24
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 24 -> 12
            nn.Conv2d(6, 16, kernel_size=5),     # 12 -> 8
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CustomCNN(nn.Module):
    """Your original CNN, cleaned up for comparison."""
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
print(f"Image size             : 28x28 grayscale")
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
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, str(val), ha="center", va="bottom", color="white", fontsize=8)
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
    composite[r*28:(r+1)*28, c*28:(c+1)*28] = mean_img
ax3.imshow(composite, cmap="magma", interpolation="nearest")
ax3.axis("off")
for idx, lbl in enumerate(raw_unique_labels):
    r, c = divmod(idx, n_cols)
    ax3.text(c*28 + 14, r*28 + 26, DISPLAY_LABEL_MAP[lbl], ha="center", va="bottom", color="white", fontsize=7, fontweight="bold")

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
        sample_grid[ri*28:(ri+1)*28, ci*28:(ci+1)*28] = row.reshape(28, 28)
ax5.imshow(sample_grid, cmap="gray", interpolation="nearest")
ax5.axis("off")
for ri, lbl in enumerate(show_labels):
    ax5.text(-3, ri*28 + 14, DISPLAY_LABEL_MAP[lbl], ha="right", va="center", color="#e040fb", fontsize=12, fontweight="bold")

fig.suptitle("ASL Sign Language MNIST — Exploratory Data Analysis", color="white", fontsize=20, fontweight="bold", y=0.98)
plt.savefig(f"{OUT_DIR}/1_eda.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print(f"EDA saved -> {OUT_DIR}/1_eda.png")


# =============================================================
# 5. TRAINING UTILITIES
# =============================================================
def create_optimizer_and_scheduler(model, scheduler_factor=0.5):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=scheduler_factor
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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_labels, all_preds


def train_model(model_name, model, scheduler_factor=0.5):
    print("\n" + "-" * 70)
    print(f"Training {model_name}")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(model, scheduler_factor=scheduler_factor)

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

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | "
            f"val loss={va_loss:.4f}, acc={va_acc*100:.2f}%"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            print(f"  New best validation accuracy: {best_val_acc*100:.2f}%")
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

    print(f"\n{model_name} test accuracy: {test_acc*100:.2f}%")
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
        "scheduler_factor": scheduler_factor
    }


# =============================================================
# 6. TRAIN BOTH MODELS
# =============================================================
print("\n" + "=" * 70)
print("3 · TRAINING MODELS (LENET-5 + CUSTOM CNN SCHEDULER TESTS)")
print("=" * 70)

lenet_results = train_model(
    "LeNet5", LeNet5(num_classes=num_classes).to(DEVICE), scheduler_factor=0.5
)
custom_results_05 = train_model(
    "CustomCNN_f0_5", CustomCNN(num_classes=num_classes).to(DEVICE), scheduler_factor=0.5
)
custom_results_01 = train_model(
    "CustomCNN_f0_1", CustomCNN(num_classes=num_classes).to(DEVICE), scheduler_factor=0.1
)
results = [lenet_results, custom_results_05, custom_results_01]


# =============================================================
# 7. PLOTS FOR BOTH MODELS
# =============================================================
print("\n" + "=" * 70)
print("4 · SAVING COMPARISON PLOTS")
print("=" * 70)

# Learning curves comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax in axes:
    ax.set_facecolor("#0f0f1a")
fig.patch.set_facecolor("#0f0f1a")

for res in results:
    epochs = range(1, len(res["history"]["train_loss"]) + 1)
    axes[0].plot(epochs, res["history"]["train_loss"], linewidth=2, label=f"{res['model_name']} Train")
    axes[0].plot(epochs, res["history"]["val_loss"], linestyle="--", linewidth=2, label=f"{res['model_name']} Val")
    axes[1].plot(epochs, res["history"]["train_acc"], linewidth=2, label=f"{res['model_name']} Train")
    axes[1].plot(epochs, res["history"]["val_acc"], linestyle="--", linewidth=2, label=f"{res['model_name']} Val")

axes[0].set_title("Loss Comparison", color="white", fontsize=14, fontweight="bold")
axes[1].set_title("Accuracy Comparison", color="white", fontsize=14, fontweight="bold")
for ax in axes:
    ax.set_xlabel("Epoch", color="#aaaacc")
    ax.tick_params(colors="white")
    ax.grid(color="#222244", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
fig.suptitle("Training History Comparison", color="white", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_model_comparison_curves.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()

# Confusion matrices
for res in results:
    fig_cm, ax = plt.subplots(figsize=(14, 12))
    fig_cm.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    cm = confusion_matrix(res["all_labels"], res["all_preds"], labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="magma",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    ax.set_title(f"{res['model_name']} — Normalised Confusion Matrix (Test Acc = {res['test_acc']*100:.1f}%)",
                 color="white", fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted", color="#aaaacc", fontsize=12)
    ax.set_ylabel("True", color="#aaaacc", fontsize=12)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{res['model_name'].lower()}_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()

# Accuracy bar chart
fig_bar, ax = plt.subplots(figsize=(8, 6))
fig_bar.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")
model_names = [r["model_name"] for r in results]
acc_values = [r["test_acc"] * 100 for r in results]
bars = ax.bar(model_names, acc_values)
ax.set_title("Test Accuracy Comparison", color="white", fontsize=16, fontweight="bold")
ax.set_ylabel("Accuracy (%)", color="#aaaacc")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.grid(axis="y", color="#222244", linewidth=0.5)
for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.2f}%", ha="center", color="white", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_test_accuracy_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()

# =============================================================
# 8. SUMMARY TABLE
# =============================================================
summary_rows = []
for res in results:
    summary_rows.append({
        "Model": res["model_name"],
        "Parameters": res["n_params"],
        "Best Val Accuracy": round(res["best_val_acc"] * 100, 2),
        "Test Accuracy": round(res["test_acc"] * 100, 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT_DIR}/model_comparison_summary.csv", index=False)
with open(f"{OUT_DIR}/model_comparison_summary.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

print("\n" + "=" * 70)
print("5 · PIPELINE SUMMARY")
print("=" * 70)
print(f"Dataset             : Sign Language MNIST")
print(f"Image type          : 28x28 grayscale")
print(f"Train / Val / Test  : {len(train_dataset):,} / {len(val_dataset):,} / {len(test_dataset):,}")
print(f"Preprocessing       : scale to [0,1], train-only light augmentation")
print(f"Loss function       : CrossEntropyLoss")
print(f"Optimizer           : Adam")
print(f"Learning rate       : {LR}")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Epochs              : {EPOCHS}")
print("\nModel comparison:")
print(summary_df.to_string(index=False))
print(f"\nAll outputs saved in ./{OUT_DIR}/")
print("=" * 70)
