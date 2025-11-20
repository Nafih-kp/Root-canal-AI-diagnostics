from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640)

# Save the model
model.save('dental_yolo.pt')

# Export to ONNX for web use
model.export(format='onnx')