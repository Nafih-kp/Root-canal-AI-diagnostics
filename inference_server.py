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
from contourlet_filter import ContourletTransform

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = r'runs\detect\train\weights\best.pt'
model = None
contourlet_filter = None
use_filter = True

def load_model():
    global model
    if model is None:
        try:
            model = YOLO(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    return True

def load_filter():
    global contourlet_filter
    if contourlet_filter is None:
        try:
            contourlet_filter = ContourletTransform(num_levels=2, num_directions=8)
            print("✓ Contourlet filter initialized")
        except Exception as e:
            print(f"⚠️  Error initializing filter: {e}")
            return False
    return True

def apply_preprocessing(img_array):
    """Apply Contourlet transform preprocessing"""
    if not use_filter:
        return img_array
    
    if contourlet_filter is None:
        return img_array
    
    try:
        filtered = contourlet_filter.apply(img_array)
        return filtered
    except Exception as e:
        print(f"⚠️  Error applying filter: {e}")
        return img_array

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if model is not None else "not_loaded"
    filter_status = "active" if (use_filter and contourlet_filter is not None) else "disabled"
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "filter": filter_status
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Load model if not loaded
        if not load_model():
            return jsonify({"error": "Model failed to load"}), 500
        
        # Load filter if using filtering
        if use_filter and not load_filter():
            print("⚠️  Filter initialization failed, continuing without filter")

        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Contourlet preprocessing
        img_array = apply_preprocessing(img_array)

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
    print("=" * 60)
    print("Dental X-Ray Detection Server")
    print("=" * 60)
    print("\nInitializing components...")
    
    if load_model():
        print(f"✓ Model loaded from: {model_path}")
    else:
        print("⚠️  Model loading deferred (will try on first request)")
    
    if use_filter and load_filter():
        print("✓ Contourlet filter enabled")
    else:
        print("⚠️  Running without Contourlet filter")
    
    print(f"\nServer configuration:")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5000")
    print(f"  Filter enabled: {use_filter}")
    print(f"\nServer starting...")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print("  GET  /health  - Health check")
    print("  POST /detect  - Run detection on uploaded image")
    print("\n" + "=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)