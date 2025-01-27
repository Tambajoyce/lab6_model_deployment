from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import joblib

# Load the saved model
model_filename = 'C:/Users/PC/Desktop/DEGREE/LABS/lab 6/hdbscan_face_recognition_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = joblib.load(file)

print("Model loaded successfully!")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Load the image
    image = Image.open(request.files['image']).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = loaded_model(image)
        _, predicted = outputs.max(1)  # Get the class index
        class_index = predicted.item()

    return jsonify({'predicted_class': class_index})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)