# XrayAI

An AI-powered chest X-ray disease detection system built with **ResNet-50**, **PyTorch**, **FastAPI**, and **Grad-CAM**.

XrayAI analyzes a chest X-ray image and predicts one of four classes:

- COVID-19
- Lung Opacity
- Normal
- Viral Pneumonia

The system also generates a **Grad-CAM heatmap** to visually highlight the regions that contributed to the model's prediction.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## Features

- 🩻 Chest X-ray image classification
- 🤖 ResNet-50 deep learning model
- 🔥 Grad-CAM visual explanations
- ⚡ FastAPI backend
- 🌐 Web-based interface
- 📊 Prediction confidence and class probabilities
- 🖥️ Desktop launcher using PyWebView
- 📱 Responsive web interface
- 💾 PyTorch model stored using Git LFS

---

## Model Performance

The trained model was evaluated on a held-out test set containing **3,176 images**.

### Overall Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **97.01%** |
| Macro F1-Score | **97.60%** |
| Weighted F1-Score | **97.01%** |

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| COVID | 98.59% | 99.12% | 98.85% |
| Lung Opacity | 94.09% | 96.52% | 95.29% |
| Normal | 97.94% | 96.21% | 97.07% |
| Viral Pneumonia | 98.95% | 99.47% | 99.21% |

These results represent performance on this project's test split and should not be interpreted as clinical validation.

---

## Technology Stack

### Machine Learning

- Python 3.11
- PyTorch
- Torchvision
- ResNet-50
- Transfer Learning
- Grad-CAM
- Scikit-learn
- Pillow
- NumPy

### Backend

- FastAPI
- Uvicorn
- Python Multipart

### Frontend

- HTML5
- CSS3
- JavaScript

### Desktop

- PyWebView

### Version Control

- Git
- GitHub
- Git LFS

---

## Project Architecture

```text
                 ┌─────────────────────┐
                 │     User uploads    │
                 │     X-ray image     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    FastAPI Backend   │
                 │      /predict       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Image Processing  │
                 │ Resize + Normalize  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      ResNet-50      │
                 │    PyTorch Model    │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐   ┌─────────────────┐
        │   Prediction    │   │    Grad-CAM     │
        │ + Probabilities │   │    Heatmap      │
        └────────┬────────┘   └────────┬────────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │    Web Interface    │
                 │ Result + Heatmap    │
                 └─────────────────────┘
```

---

## 📁 Project Structure

```text
XrayAI/
├── app/
│   ├── backend.py
│   ├── index.html
│   └── assets/
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── app.js
│           └── graphics.js
│
├── checkpoints/
│   └── best_model.pth
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── icon.icns
├── icon.png
├── launcher.py
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

The X-ray dataset is not included in this repository because of its large size and dataset licensing considerations.

---

## How XrayAI Works

### 1. Upload X-ray

The user uploads a chest X-ray image through the web interface.

### 2. Image Preprocessing

The uploaded image is processed before being passed to the model:

- Resized to **224 × 224 pixels**
- Converted to 3-channel format
- Converted to a PyTorch tensor
- Normalized using ImageNet normalization values

### 3. AI Prediction

The processed image is passed through the fine-tuned **ResNet-50** model.

The model produces probabilities for all four classes.

### 4. Classification

The class with the highest probability is selected as the predicted class.

### 5. Grad-CAM Explanation

Grad-CAM analyzes the model's internal activations and generates a heatmap showing the regions of the X-ray that contributed most to the prediction.

### 6. Result Display

The web interface displays:

- Predicted class
- Confidence score
- Class probabilities
- Grad-CAM heatmap

---

## Model

XrayAI uses **ResNet-50**, a convolutional neural network architecture originally trained on ImageNet.

Transfer learning was used instead of training the entire network from scratch.

The final classification layer was modified for the project's four classes:

```text
COVID
Lung_Opacity
Normal
Viral Pneumonia
```

The training pipeline also uses:

- Image augmentation
- Class-weighted Cross Entropy Loss
- Adam optimizer
- Learning-rate scheduling
- Validation-based best-model selection

---

## Grad-CAM

XrayAI uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations.

Instead of only displaying a prediction such as:

```text
Prediction: Lung Opacity
Confidence: 99.9%
```

the system also generates a heatmap showing the image regions that influenced the model's prediction.

This makes the model's output more interpretable and demonstrates an explainable-AI component.

---

## Installation

### Requirements

- Python 3.11
- Git
- Git LFS

### Clone the Repository

```bash
git clone https://github.com/brahmjotsingh02/XrayAI.git
```

Enter the project directory:

```bash
cd XrayAI
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start XrayAI using:

```bash
python launcher.py
```

The launcher starts the FastAPI backend and opens the XrayAI interface.

The web application is available at:

```text
http://localhost:8000
```

---

## API

### Prediction Endpoint

```text
POST /predict
```

The endpoint accepts a chest X-ray image and returns:

- Predicted class
- Confidence score
- Class probabilities
- Grad-CAM heatmap

### Example Response

```json
{
  "prediction": "Lung_Opacity",
  "confidence": 0.999,
  "probabilities": {},
  "heatmap": "base64-image-data"
}
```

---

## Training

The training pipeline can be executed using:

```bash
python src/train.py
```

The training process includes:

1. Dataset loading
2. Train/validation/test splitting
3. Image augmentation
4. Image normalization
5. ResNet-50 initialization
6. Transfer learning
7. Class-weighted loss
8. Model optimization
9. Validation evaluation
10. Best-model checkpoint saving

The trained model is saved to:

```text
checkpoints/best_model.pth
```

---

## Model Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py
```

The evaluation script provides:

- Test accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

---

## Dataset

The project uses the **COVID-19 Radiography Database** with four selected categories:

- COVID
- Lung Opacity
- Normal
- Viral Pneumonia

The dataset is intentionally **not included in this GitHub repository** because of its large size and dataset licensing considerations.

---

## Git LFS

The trained model is approximately **90 MB**, so Git LFS is used to store:

```text
checkpoints/best_model.pth
```

After cloning the repository, make sure Git LFS is installed:

```bash
git lfs install
```

Then pull the model:

```bash
git lfs pull
```

---

## Desktop Application

XrayAI also includes a PyWebView-based desktop launcher.

Run:

```bash
python launcher.py
```

The launcher starts the FastAPI server and opens the application in a desktop window.

---

## Current Results

The current trained model achieved:

**97.01% test accuracy**

with a:

**97.60% macro F1-score**

The strongest individual performance was observed for **Viral Pneumonia**, while the main classification confusion occurred between **Lung Opacity** and **Normal** classes.

---

## Future Improvements

Potential future improvements include:

- Deploying the application as a public web service
- Adding more chest X-ray disease categories
- Improving model calibration
- Adding automated testing
- Adding model versioning
- Improving mobile UI
- Adding user authentication
- Adding cloud-based model storage
- Improving deployment scalability
- Performing additional validation on independent datasets

---

## Limitations

- The model is trained and evaluated on a specific dataset.
- Test-set performance does not guarantee performance on real-world clinical data.
- Dataset bias may affect predictions.
- The model should not be used for medical diagnosis.
- Additional external validation would be required before any clinical application.

---

## Author

**Brahmjot Singh**

GitHub:  
https://github.com/brahmjotsingh02

---

## License

This project is intended for **educational and research purposes**.

It is not intended to provide medical diagnosis, treatment recommendations, or professional medical advice.