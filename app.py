import os
import traceback

import gdown
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_resnet18_pharyngitis.pth")
HTML_PATH = os.path.join(BASE_DIR, "pharyscan_web_app.html")

MODEL_URL = "https://drive.google.com/uc?id=1nhaLsnCFmxj-WtnEPM6haATWY9VXVp2L"

CLASS_NAMES = ["no_pharyngitis", "pharyngitis"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Download model
# -----------------------------
def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        print("Model already exists.")
        return

    print("Model not found. Downloading from Google Drive...")

    downloaded = gdown.download(
        MODEL_URL,
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

    if downloaded is None:
        raise RuntimeError("Model download failed. gdown returned None.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file was not created after download.")

    if os.path.getsize(MODEL_PATH) == 0:
        raise RuntimeError("Downloaded model file is empty.")

    print("Model downloaded successfully.")


# -----------------------------
# Load model
# -----------------------------
def load_model():
    global model

    download_model_if_needed()

    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        cleaned_checkpoint = {}

        for key, value in checkpoint.items():
            new_key = key.replace("module.", "")
            cleaned_checkpoint[new_key] = value

        net.load_state_dict(cleaned_checkpoint)
        model = net

    else:
        model = checkpoint

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}.")


# -----------------------------
# Error handlers
# -----------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    if not os.path.exists(HTML_PATH):
        return jsonify({
            "success": False,
            "error": "pharyscan_web_app.html file not found"
        }), 500

    return send_file(HTML_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@app.route("/analyze", methods=["POST"])
@app.route("/upload", methods=["POST"])
@app.route("/infer", methods=["POST"])
@app.route("/run_inference", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model is not loaded. Check Render logs."
            }), 500

        if not request.files:
            return jsonify({
                "success": False,
                "error": "No image file uploaded"
            }), 400

        file = next(iter(request.files.values()))

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image file selected"
            }), 400

        try:
            image = Image.open(file.stream).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({
                "success": False,
                "error": "Uploaded file is not a valid image"
            }), 400

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]

        confidence_score = round(confidence.item() * 100, 2)
        no_pharyngitis_score = round(probabilities[0].item() * 100, 2)
        pharyngitis_score = round(probabilities[1].item() * 100, 2)

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "class": predicted_class,
            "label": predicted_class,
            "result": predicted_class,
            "confidence": confidence_score,
            "confidence_score": confidence_score,
            "no_pharyngitis": no_pharyngitis_score,
            "pharyngitis": pharyngitis_score,
            "probabilities": {
                "no_pharyngitis": no_pharyngitis_score,
                "pharyngitis": pharyngitis_score
            }
        }), 200

    except Exception as e:
        print("Prediction error:")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------
# Load model when server starts
# -----------------------------
try:
    load_model()
except Exception:
    print("Model loading failed:")
    traceback.print_exc()
    model = None


# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
