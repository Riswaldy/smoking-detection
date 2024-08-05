import torch
from ultralytics import YOLO
from roboflow import Roboflow

# Initialize Roboflow and download the dataset
rf = Roboflow(api_key="fgYhT92PQ1vVHHdPHoS7")
project = rf.workspace("comvis-nojup").project("smoking-detection-2sjky")
version = project.version(1)
dataset = version.download("yolov8", location="C:/Users/user/Desktop/and/final/smoking-detection")


# Initialize YOLOv8 model for training
model = YOLO('yolov8n.yaml')  # or any other YOLOv8 model configuration

# Train the model
results = model.train(data="C:/Users/user/Desktop/and/final/smoking-detection/data.yaml", epochs=50, imgsz=640)

# Check if training results contain the checkpoint
if hasattr(model, 'ckpt') and model.ckpt:
    # Save the trained model
    model.save('yolov8_smoking_detection.pt')
else:
    print("Training did not produce a checkpoint. Please check the training process.")
