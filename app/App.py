from ultralytics import YOLO
import cv2


class processor:

    # Load a YOLO11n PyTorch model
    model = YOLO("yolo11n-seg.pt")

    # Export the model to NCNN format
    model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

    # Load the exported NCNN model
    ncnn_model = YOLO("yolo11n_ncnn_model", task="segment")




    def ProcessFrame(self,location, name1):
        results = self.ncnn_model(location)
        annotatedresults = results[0].plot()
        cv2.imwrite(name1+".jpg",annotatedresults)
        r = results[0]
                # Iterate through detected boxes
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()      # [x1, y1, x2, y2]
            conf = box.conf[0].item()             # confidence score
            cls = int(box.cls[0].item())          # class ID
            class_name = self.model.names[cls]
            print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")



if __name__ == "__main__":
    processer1 = processor()

    terminalstring = ""
    while(terminalstring != "stop"):
        terminalstring = input()
        print(terminalstring)
        name2 = input()
        processer1.ProcessFrame(terminalstring, name2)

