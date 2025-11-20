from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image
import ultralytics
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = r'runs\detect\train\weights\best.pt'
model = None

def load_model():
    global model
    if model is None:
        try:
            model = YOLO(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Load model if not loaded
        if not load_model():
            return jsonify({"error": "Model failed to load"}), 500

        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        # Convert to numpy array
        img_array = np.array(image)

        # Run inference
        results = model(img_array)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Convert to normalized coordinates
                img_height, img_width = img_array.shape[:2]
                x = x1 / img_width
                y = y1 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Map class ID to name
                class_names = ['No Endodontic Treatment', 'Incomplete Endodontic Treatment',
                             'Complete Endodontic Treatment', 'Total Endodontic Failure']
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                detections.append({
                    'bbox': [x, y, width, height],
                    'class': class_name,
                    'score': confidence
                })

        return jsonify({"detections": detections})

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting inference server...")
    app.run(host='0.0.0.0', port=5000, debug=True)