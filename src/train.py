import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from dataset import get_dataloaders, get_class_weights, CLASSES


# ============================================================
# Configuration
# ============================================================

EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Project root = folder containing src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAVE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print(f"Using device: {DEVICE}")
print(f"Model will be saved to: {SAVE_PATH}")


# ============================================================
# Data
# ============================================================

train_loader, val_loader, test_loader = get_dataloaders(
    BATCH_SIZE
)


# ============================================================
# Model
# ============================================================

model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V1
)


# Freeze the pretrained layers
for param in model.parameters():
    param.requires_grad = False


# Fine-tune ResNet-50 layer4
for param in model.layer4.parameters():
    param.requires_grad = True


# Replace the original ImageNet classifier
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(
        model.fc.in_features,
        len(CLASSES)
    )
)


model = model.to(DEVICE)


# ============================================================
# Loss function
# ============================================================

weights = get_class_weights().to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Train / Validation epoch
# ============================================================

def run_epoch(loader, training=True):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()
                optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# Training
# ============================================================

SAVE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

best_val_loss = float("inf")


print("\nStarting training...")
print(f"Classes: {CLASSES}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print()


for epoch in range(
    1,
    EPOCHS + 1
):

    start_time = time.time()

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    scheduler.step(val_loss)

    elapsed = (
        time.time()
        - start_time
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} "
        f"Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} "
        f"Acc: {val_acc:.3f} | "
        f"LR: {current_lr:.6f} | "
        f"{elapsed:.1f}s"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print(
            f"  ✓ Saved best model "
            f"(val_loss={val_loss:.4f})"
        )


print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved at: {SAVE_PATH}")