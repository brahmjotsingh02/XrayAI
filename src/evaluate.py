from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from dataset import get_dataloaders, CLASSES


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAVE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)


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
print(f"Model path: {SAVE_PATH}")


# ============================================================
# Validate model
# ============================================================

if not SAVE_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {SAVE_PATH}"
    )


# ============================================================
# Test data
# ============================================================

_, _, test_loader = get_dataloaders(
    batch_size=32
)


# ============================================================
# Load model
# ============================================================

model = models.resnet50(
    weights=None
)

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(
        model.fc.in_features,
        len(CLASSES)
    )
)

model.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# Evaluate
# ============================================================

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = (
            outputs
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# Results
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n" + "=" * 60)
print("XrayAI Model Evaluation")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


print("\n── Classification Report ──")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)


print("── Confusion Matrix ──")

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print(matrix)


print("\nClass order:")
for index, class_name in enumerate(CLASSES):
    print(f"{index}: {class_name}")

print("\nEvaluation complete.")