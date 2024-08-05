import torch
import ultralytics
import cv2
import os
from ultralytics import YOLO
import numpy as np

# Verify the current working directory

# Load the model
model = YOLO('runs/detect/train/weights/best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()


    if success:
        # Convert the frame to a tensor (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model(frame_tensor)

        # Annotate the original BGR frame with the detection results
        annotated_frame = frame.copy()  # Copy the original frame to annotate

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw the bounding box on the original BGR frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Prepare the label for the bounding box
                class_name = model.names[int(box.cls)]
                label = f"{class_name}: {box.conf.item():.2f}"
                # Put the label on the frame
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
