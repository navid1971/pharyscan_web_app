import os
import gdown
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "best_resnet18_pharyngitis.pth"
HTML_PATH = "pharyscan_web_app.html"

# Direct Google Drive download link
MODEL_URL = "https://drive.google.com/uc?id=1nhaLsnCFmxj-WtnEPM6haATWY9VXVp2L"

CLASS_NAMES = ["no_pharyngitis", "pharyngitis"]

# -----------------------------
# Download model from Google Drive
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model download failed. Check Google Drive sharing settings."
        )

    print("Model downloaded successfully.")

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load ResNet18 model
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict):
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        new_checkpoint = {}
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")
            new_checkpoint[new_key] = v
        model.load_state_dict(new_checkpoint)
else:
    model = checkpoint

model.to(device)
model.eval()

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
# Routes
# -----------------------------
@app.route("/")
def home():
    return send_file(HTML_PATH)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@app.route("/analyze", methods=["POST"])
@app.route("/upload", methods=["POST"])
@app.route("/infer", methods=["POST"])
@app.route("/run_inference", methods=["POST"])
def predict():
    try:
        if not request.files:
            return jsonify({
                "success": False,
                "error": "No image file uploaded"
            }), 400

        # Accept any frontend file field name: image, file, upload, etc.
        file = next(iter(request.files.values()))

        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = round(confidence.item() * 100, 2)

        no_phar_score = round(probabilities[0].item() * 100, 2)
        phar_score = round(probabilities[1].item() * 100, 2)

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "class": predicted_class,
            "label": predicted_class,
            "result": predicted_class,
            "confidence": confidence_score,
            "confidence_score": confidence_score,
            "no_pharyngitis": no_phar_score,
            "pharyngitis": phar_score,
            "probabilities": {
                "no_pharyngitis": no_phar_score,
                "pharyngitis": phar_score
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
