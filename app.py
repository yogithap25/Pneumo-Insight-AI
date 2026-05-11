from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import time
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained.h5") 
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Ensure the static folder exists
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Preprocess image function
def preprocess_image(image_path, target_size=(300, 300)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img, img_array

# Function to generate LIME explanation
def generate_lime_explanation(image_array):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_array,
        model.predict,
        top_labels=1,          # Binary classification
        hide_color=0,          # Hide color for occluded regions
        num_samples=200        # Number of perturbed samples
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],  # Get the predicted class
        positive_only=True,         # Show only regions contributing to prediction
        num_features=5,             # Number of regions to highlight
        hide_rest=False            # Show the rest of the image
    )
    # Convert to uint8 and scale
    temp = (temp * 255).astype(np.uint8)
    masked_img = mark_boundaries(temp, mask)
    return masked_img

def lime_predict(images):
    images = np.array(images)
    preds = model.predict(images)
    return np.concatenate([1 - preds, preds], axis=1)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    unique_filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(STATIC_DIR, unique_filename)

    try:
        file.save(file_path)

        original_image, image_array = preprocess_image(file_path)

        prediction = model.predict(np.expand_dims(image_array, axis=0))
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if predicted_class == "Pneumonia" else 1 - prediction[0][0]

        lime_explanation = generate_lime_explanation(image_array)

        lime_filename = str(uuid.uuid4()) + "_lime.jpg"
        lime_path = os.path.join(STATIC_DIR, lime_filename)

        # ✅ FIXED saving
        plt.imsave(lime_path, np.clip(lime_explanation, 0, 1))

        timestamp = int(time.time())

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence),
            "image_url": f"/static/{unique_filename}?{timestamp}",
            "lime_url": f"/static/{lime_filename}?{timestamp}"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)