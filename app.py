import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

MODEL_PATH = "best_resnet18_pharyngitis.pth"
HTML_PATH = "pharyscan_web_app.html"

CLASS_NAMES = ["no_pharyngitis", "pharyngitis"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

checkpoint = torch.load(MODEL_PATH, map_location=device)

# If model was saved as state_dict
if isinstance(checkpoint, dict):
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # Fix for models saved with "module." prefix
        new_checkpoint = {}
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")
            new_checkpoint[new_key] = v
        model.load_state_dict(new_checkpoint)
else:
    # If full model was saved
    model = checkpoint

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/")
def home():
    return send_file(HTML_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" in request.files:
            file = request.files["image"]
        elif "file" in request.files:
            file = request.files["file"]
        else:
            return jsonify({"error": "No image file uploaded"}), 400

        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = round(confidence.item() * 100, 2)

        return jsonify({
            "prediction": predicted_class,
            "class": predicted_class,
            "label": predicted_class,
            "confidence": confidence_score,
            "confidence_score": confidence_score,
            "probabilities": {
                CLASS_NAMES[0]: round(probabilities[0].item() * 100, 2),
                CLASS_NAMES[1]: round(probabilities[1].item() * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    return predict()

@app.route("/analyze", methods=["POST"])
def analyze():
    return predict()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
