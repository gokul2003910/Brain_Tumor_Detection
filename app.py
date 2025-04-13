from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
MODEL_PATH = "D:\\Brain-Tumor-Detection\\model\\Model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Get model input size dynamically
input_shape = model.input_shape[1:3]  

# Class labels
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return "No selected file", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=input_shape)
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  
    result = tumor_classes[predicted_class]  

    return render_template("index.html", result=result, uploaded_image=filepath)

if __name__ == "__main__":
    app.run(debug=True)
