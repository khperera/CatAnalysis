from ultralytics import YOLO
import cv2

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo11n_ncnn_model")

# Run inference
results = ncnn_model("./bunt1.jpg")

annotatedresults = results[0].plot()
cv2.imwrite("hi.jpg",annotatedresults)

r = results[0]

# Iterate through detected boxes
for box in r.boxes:
    xyxy = box.xyxy[0].cpu().numpy()      # [x1, y1, x2, y2]
    conf = box.conf[0].item()             # confidence score
    cls = int(box.cls[0].item())          # class ID
    class_name = model.names[cls]
    print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")