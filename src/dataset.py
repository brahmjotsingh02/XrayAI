import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path


# ============================================================
# Dataset location
# ============================================================

# Project root = parent folder of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.environ.get(
        "XRAY_DATA_DIR",
        PROJECT_ROOT / "data" / "COVID-19_Radiography_Dataset"
    )
).expanduser()


# ============================================================
# Classes
# ============================================================

CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}


# ============================================================
# Image transforms
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    # Data augmentation - training only
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Dataset
# ============================================================

class XrayDataset(Dataset):

    def __init__(self, data_dir, classes, transform=None):
        self.transform = transform
        self.samples = []

        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {data_dir}\n"
                "Set the XRAY_DATA_DIR environment variable to the "
                "correct location of the COVID-19_Radiography_Dataset "
                "folder, or move the dataset to the default path."
            )

        for class_name in classes:
            img_dir = data_dir / class_name / "images"

            if not img_dir.exists():
                print(
                    f"Warning: image directory not found: {img_dir}"
                )
                continue

            for img_path in img_dir.glob("*.png"):
                self.samples.append(
                    (img_path, CLASS_TO_IDX[class_name])
                )

        if not self.samples:
            raise RuntimeError(
                f"No PNG images found in dataset directory: {data_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        with Image.open(img_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image, label


# ============================================================
# Train / Validation / Test split
# ============================================================

def get_dataloaders(batch_size=32):

    # Create dataset WITHOUT a transform.
    # Individual subsets will receive their own transforms.
    base_dataset = XrayDataset(
        DATA_DIR,
        CLASSES,
        transform=None
    )

    total = len(base_dataset)

    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    # Reproducible split
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        total,
        generator=generator
    ).tolist()

    train_indices = indices[:train_size]

    val_indices = indices[
        train_size:train_size + val_size
    ]

    test_indices = indices[
        train_size + val_size:
    ]

    # Create THREE independent dataset objects.
    # This prevents the transform-sharing bug.
    train_dataset = XrayDataset(
        DATA_DIR,
        CLASSES,
        transform=train_transforms
    )

    val_dataset = XrayDataset(
        DATA_DIR,
        CLASSES,
        transform=val_transforms
    )

    test_dataset = XrayDataset(
        DATA_DIR,
        CLASSES,
        transform=val_transforms
    )

    train_set = Subset(
        train_dataset,
        train_indices
    )

    val_set = Subset(
        val_dataset,
        val_indices
    )

    test_set = Subset(
        test_dataset,
        test_indices
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Train: {len(train_set)} | "
        f"Val: {len(val_set)} | "
        f"Test: {len(test_set)}"
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Class weights
# ============================================================

def get_class_weights():

    counts = [
        3616,   # COVID
        6012,   # Lung_Opacity
        10192,  # Normal
        1345    # Viral Pneumonia
    ]

    total = sum(counts)

    weights = [
        total / (len(counts) * count)
        for count in counts
    ]

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# Test dataset pipeline
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Class weights: {get_class_weights()}")