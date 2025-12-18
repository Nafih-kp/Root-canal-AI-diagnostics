
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
from grad_cam import YOLOGradCAM, overlay_heatmap
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

config = {}
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
except:
    print("⚠️  model_config.json not found, using defaults")

default_model_name = config.get('default_model', 'student_distilled')
model_specs = config.get('available_models', {})
model_path = model_specs.get(default_model_name, {}).get('path', 'dental_yolo_roboflow_filtered.pt')
use_filter = config.get('preprocessing', {}).get('contourlet_enabled', True)

model = None
contourlet_filter = None

class_names = [
    'No Endodontic Treatment',
    'Complete Endodontic Treatment',
    'Incomplete Endodontic Treatment',
    'Total Endodontic Failure'
]

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
        if img_array.dtype != np.float32 and img_array.dtype != np.float64:
            img_array = img_array.astype(np.float32) / 255.0
        
        filtered = contourlet_filter.apply(img_array)
        
        if filtered is None:
            print("⚠️  Contourlet filter returned None, using original image")
            return img_array
            
        return filtered
    except Exception as e:
        print(f"⚠️  Error applying filter: {e}")
        return img_array

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if model is not None else "not_loaded"
    filter_status = "active" if (use_filter and contourlet_filter is not None) else "disabled"
    
    health_data = {
        "status": "healthy",
        "model": {
            "status": model_status,
            "name": default_model_name,
            "path": model_path
        },
        "filter": {
            "status": filter_status,
            "enabled": use_filter
        },
        "available_models": list(model_specs.keys())
    }
    
    if default_model_name in model_specs:
        health_data["model"]["info"] = model_specs[default_model_name]
    
    return jsonify(health_data)

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        "available_models": model_specs,
        "current_model": default_model_name
    })

@app.route('/models/<model_name>', methods=['POST'])
def switch_model(model_name):
    global model, model_path, use_filter, default_model_name
    
    if model_name not in model_specs:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    
    new_model_path = model_specs[model_name].get('path')
    if not new_model_path or not Path(new_model_path).exists():
        return jsonify({"error": f"Model file not found: {new_model_path}"}), 404
    
    try:
        model = None
        model_path = new_model_path
        default_model_name = model_name
        
        if not load_model():
            return jsonify({"error": "Failed to load new model"}), 500
        
        if use_filter and not load_filter():
            print("⚠️  Filter initialization failed, continuing without filter")
        
        return jsonify({
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "model_info": model_specs[model_name]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

        # Decode base64 image with better error handling
        try:
            base64_string = data['image']
            
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            print(f"✓ Image decoded: format={image.format}, size={image.size}, mode={image.mode}")
        except (ValueError, IndexError) as e:
            print(f"✗ Base64 decoding error: {e}")
            return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
        except Exception as e:
            print(f"✗ Image parsing error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to parse image: {str(e)}"}), 400

        # Convert to numpy array and ensure proper format
        try:
            img_array = np.array(image)
            
            if img_array.size == 0:
                return jsonify({"error": "Image is empty"}), 400
            
            if len(img_array.shape) == 2:
                print(f"⚠️  Grayscale image detected, converting to BGR")
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    print(f"⚠️  RGBA image detected, converting to BGR")
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:
                    pass
                
            print(f"✓ Image loaded: shape={img_array.shape}, dtype={img_array.dtype}")
        except Exception as e:
            print(f"✗ Image conversion error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to convert image to array: {str(e)}"}), 400
        
        # Apply Contourlet preprocessing
        try:
            img_array = apply_preprocessing(img_array)
        except Exception as e:
            print(f"✗ Preprocessing error: {e}")
            return jsonify({"error": f"Failed during preprocessing: {str(e)}"}), 500

        # Run inference
        try:
            print(f"Running inference on image shape: {img_array.shape}")
            results = model.predict(img_array, conf=0.25)
            print(f"✓ Inference completed, processing results")
        except Exception as e:
            print(f"✗ Inference error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"YOLO inference failed: {str(e)}"}), 500
        
        # --- GRAD-CAM GENERATION ---
        heatmap_base64 = None
        try:
            # Check if user requested explainability
            # data was loaded earlier from request.get_json()
            raw_enable_gradcam = data.get('enable_gradcam')
            use_gradcam = raw_enable_gradcam is True or str(raw_enable_gradcam).lower() == 'true'
            
            print(f"DEBUG: raw_enable_gradcam={raw_enable_gradcam}, use_gradcam={use_gradcam}")

            if use_gradcam:
                 # Initialize Grad-CAM
                # Note: creating a new hook per request is inefficient for prod, 
                # but fine for this demo. Ideally, initialize once.
                
                print("DEBUG: Initializing YOLOGradCAM...")
                # We need to preprocess the image manually for Grad-CAM input
                # YOLOv8 preprocessing: resize to 640, normalize 0-1, CHW format
                
                # Use Ultralytics internal transform if possible, or manual
                target_layers = [model.model.model[-2]] # Default fallback
                try:
                    grad_cam = YOLOGradCAM(model, target_layers[0])
                    print("DEBUG: YOLOGradCAM initialized.")
                except Exception as e:
                    print(f"DEBUG: Failed to init YOLOGradCAM: {e}")
                    grad_cam = None

                if grad_cam:
                    # Resize for model input (640x640 is standard for YOLOv8)
                    img_resized = cv2.resize(img_array, (640, 640))
                    # Normalization handled by model usually, but let's check input
                    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    input_tensor = input_tensor.to(model.device if hasattr(model, 'device') else 'cpu')

                    print("DEBUG: Generating heatmap...")
                    # Generate heatmap
                    grayscale_cam = grad_cam(input_tensor, None)
                    print(f"DEBUG: Heatmap generated. Shape: {grayscale_cam.shape if grayscale_cam is not None else 'None'}")

                    # Resize heatmap back to original image size
                    if grayscale_cam is not None:
                         # Start Overlay
                        heatmap_overlay = overlay_heatmap(img_array, grayscale_cam[0, :])
                        
                        # Convert to base64
                        _, buffer = cv2.imencode('.jpg', heatmap_overlay)
                        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                        print("DEBUG: Heatmap overlay created and encoded.")
                    else:
                        print("DEBUG: grayscale_cam is None.")
            else:
                print("DEBUG: describe_gradcam skipped.")

        except Exception as e:
            print(f"⚠ Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()

        # Process results
        detections = []
        try:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Map class ID to name
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class": class_name,
                        "class_id": cls_id
                    })
        except Exception as e:
            print(f"✗ Result processing error: {e}")
            return jsonify({"error": f"Failed to process results: {str(e)}"}), 500

        print(f"✓ Detection completed: {len(detections)} objects detected")
        return jsonify({
            "detections": detections,
            "image_size": {"width": img_array.shape[1], "height": img_array.shape[0]},
            "heatmap": heatmap_base64
        })

    except Exception as e:
        print(f"✗ Unexpected error during detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("Dental X-Ray Detection Server - Knowledge Distillation Enabled")
    print("=" * 70)
    print("\nInitializing components...")
    print(f"\nDefault Model: {default_model_name}")
    print(f"Model Path: {model_path}")
    
    if load_model():
        print(f"✓ Model loaded successfully")
    else:
        print("⚠️  Model loading deferred (will try on first request)")
    
    if use_filter and load_filter():
        print("✓ Contourlet filter enabled")
    else:
        print("⚠️  Running without preprocessing filter")
    
    print(f"\nClass mapping:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    print(f"\nAvailable Models:")
    for model_name, specs in model_specs.items():
        marker = "→" if model_name == default_model_name else " "
        print(f"  {marker} {model_name}: {specs.get('name', 'Unknown')}")
        print(f"      {specs.get('description', '')}")
    
    print(f"\nServer configuration:")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5000")
    print(f"  Filter enabled: {use_filter}")
    print(f"  CORS enabled: Yes")
    print(f"\nServer starting...")
    print("=" * 70)
    print("\nAPI Endpoints:")
    print("  GET  /health              - Health check with model info")
    print("  GET  /models              - List available models")
    print("  POST /models/<model_name> - Switch to different model")
    print("  POST /detect              - Run detection on uploaded image")
    print("\n" + "=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)