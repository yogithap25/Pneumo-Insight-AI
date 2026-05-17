from flask import Flask, render_template, request, jsonify
import os
import uuid
import time
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# -------------------- APP SETUP --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)

# -------------------- LOAD MODEL --------------------

MODEL_PATH = os.path.join(BASE_DIR, "trained.h5")
model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# -------------------- CREATE STATIC FOLDER --------------------

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# -------------------- IMAGE PREPROCESSING --------------------

def preprocess_image(image_path, target_size=(300, 300)):
    """
    Loads and preprocesses image for prediction.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0

    return img_array

# -------------------- LIME PREDICTION FUNCTION --------------------

def lime_predict(images):
    """
    Prediction function for LIME.
    """

    images = np.array(images)

    preds = model.predict(images)

    return np.concatenate([1 - preds, preds], axis=1)

# -------------------- GENERATE LIME EXPLANATION --------------------

def generate_lime_explanation(image_array):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_array,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=200
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    temp = (temp * 255).astype(np.uint8)

    explained_image = mark_boundaries(temp, mask)

    return explained_image

# -------------------- HOME ROUTE --------------------

@app.route("/")
def index():
    return render_template("index.html")

# -------------------- PREDICTION ROUTE --------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:

        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}.jpg"

        file_path = os.path.join(STATIC_DIR, unique_filename)

        # Save uploaded image
        file.save(file_path)

        # Preprocess image
        image_array = preprocess_image(file_path)

        # Model prediction
        prediction = model.predict(
            np.expand_dims(image_array, axis=0)
        )[0][0]

        # Classification
        predicted_class = (
            "Pneumonia"
            if prediction > 0.5
            else "Normal"
        )

        # Confidence calculation
        confidence = (
            prediction
            if predicted_class == "Pneumonia"
            else 1 - prediction
        )

        # Generate LIME explanation
        lime_explanation = generate_lime_explanation(image_array)

        # Save LIME image
        lime_filename = f"{uuid.uuid4()}_lime.jpg"

        lime_path = os.path.join(STATIC_DIR, lime_filename)

        plt.imsave(
            lime_path,
            np.clip(lime_explanation, 0, 1)
        )

        timestamp = int(time.time())

        # Return JSON response
        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence),
            "image_url": f"/static/{unique_filename}?{timestamp}",
            "lime_url": f"/static/{lime_filename}?{timestamp}"
        })

    except Exception as e:

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

# -------------------- RUN APP --------------------

if __name__ == "__main__":
    app.run(debug=True)