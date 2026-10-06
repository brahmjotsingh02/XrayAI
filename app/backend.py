from pathlib import Path
import base64
import io
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# Project paths
# ============================================================

if os.environ.get("XRAY_BASE_DIR"):
    BASE_DIR = Path(
        os.environ["XRAY_BASE_DIR"]
    ).resolve()
else:
    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )


# Allow importing src/dataset.py
sys.path.insert(
    0,
    str(BASE_DIR / "src")
)

from dataset import CLASSES


# ============================================================
# Configuration
# ============================================================

SAVE_PATH = (
    BASE_DIR
    / "checkpoints"
    / "best_model.pth"
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
}


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
# Validate model file
# ============================================================

if not SAVE_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {SAVE_PATH}"
    )


# ============================================================
# Load ResNet-50
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


state_dict = torch.load(
    SAVE_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="XrayAI API",
    description="AI-powered chest X-ray classification API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Static assets
# ============================================================

ASSETS_DIR = (
    BASE_DIR
    / "app"
    / "assets"
)

if not ASSETS_DIR.exists():
    raise FileNotFoundError(
        f"Assets directory not found: {ASSETS_DIR}"
    )

app.mount(
    "/assets",
    StaticFiles(
        directory=str(ASSETS_DIR)
    ),
    name="assets"
)


# ============================================================
# Prediction
# ============================================================

def predict_image(image: Image.Image):

    tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0].cpu().numpy()

    prediction_index = int(
        probabilities.argmax()
    )

    prediction = CLASSES[
        prediction_index
    ]

    confidence = (
        float(
            probabilities[
                prediction_index
            ]
        )
        * 100
    )

    all_probabilities = {
        CLASSES[index]: round(
            float(probabilities[index])
            * 100,
            1
        )
        for index in range(len(CLASSES))
    }

    return (
        prediction,
        confidence,
        all_probabilities,
        tensor
    )


# ============================================================
# Grad-CAM
# ============================================================

def get_gradcam(
    tensor: torch.Tensor,
    original_image: Image.Image
):

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=tensor
    )[0]

    image_resized = (
        original_image
        .resize((224, 224))
        .convert("RGB")
    )

    image_array = (
        np.asarray(
            image_resized
        ).astype(np.float32)
        / 255.0
    )

    visualization = show_cam_on_image(
        image_array,
        grayscale_cam,
        use_rgb=True
    )

    buffer = io.BytesIO()

    Image.fromarray(
        visualization
    ).save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# Prediction API
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was selected."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, or WEBP."
            )
        )

    # Read file
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be 10 MB or smaller."
        )

    # Open image safely
    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    try:

        (
            prediction,
            confidence,
            probabilities,
            tensor
        ) = predict_image(image)

        heatmap = get_gradcam(
            tensor,
            image
        )

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while "
                "processing the X-ray."
            )
        )

    return JSONResponse({
        "prediction": prediction,
        "confidence": round(
            confidence,
            1
        ),
        "probabilities": probabilities,
        "heatmap": heatmap
    })


# ============================================================
# Website
# ============================================================

@app.get("/")
async def root():

    html_path = (
        BASE_DIR
        / "app"
        / "index.html"
    )

    if not html_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Website files not found."
        )

    return HTMLResponse(
        html_path.read_text(
            encoding="utf-8"
        )
    )