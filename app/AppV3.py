from ultralytics import YOLO
import cv2
import numpy as np
import time
import numpy as np
from collections import Counter

import app.cat_inference as cat_inference
import threading
import app.StateTracker as State


class Processor:
    def __init__(self):
        # Load the segmentation model directly without exporting to NCNN
        # Using the original PyTorch model which fully supports segmentation
        self.model = YOLO("yolo11m-seg.pt")
        self.current_frame = None
        self.RunningYolo = False
        
        # basically, how often frames before we run yolo.
        self.YOLOlimit = 6
        self.YOLOCounter = 0
        


        self.results = None


        self.focusedmodel = cat_inference.CatInference("./cat_classifier_mobilenet_v3.pth")

        self.bunt_counter = 0
        # self.bunt_names = ["stinky", "dirty motherfucker", "bingo boy"]
        self.bunt_names = ["stinky", "dirty motherfucker", "bingo boy", "evil guy doing evil things", "suck guy", 
                   "cheese bandit", "sock thief", "chaos goblin", "menace to society", "garbage wizard", 
                   "criminal mastermind", "bologna destroyer", "couch assassin", "tuna terrorist", 
                   "box infiltrator", "midnight marauder", "cable murderer", "keyboard walker", 
                   "curtain climber", "treat slurper", "sleep destroyer", 
                   "3am sucker","3am sucker"]
        #counter 
        self.bunt_namecountover= 60
        self.bunt_max = len(self.bunt_names)-1

        # Note: We're not exporting to NCNN format since it may not fully
        # support the segmentation prototype layers in the same way

   

  
    
    def process_mask(self, mask, imglocation, savename):
        # Read the image  
        img = cv2.imread(imglocation)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        # Ensure mask is binary (0 or 1) before scaling
        normalized_mask = np.where(mask > 0, 1.0, 0.0)
        
        # Convert to uint8 (0 or 255)
        mask_uint8 = (normalized_mask * 255).astype(np.uint8)
        
        # Apply the mask
        masked_img = cv2.bitwise_and(img, img, mask=mask_uint8)
        
        # Save the result
        cv2.imwrite(savename, masked_img)


    def process_yolo_segment(self):
        if(self.RunningYolo):
            return
        self.RunningYolo = True
        self.results = self.model(self.current_frame, task="segment", verbose=False)
        self.RunningYolo = False

    def get_bunt_name(self):
        pass

        
    def process_frame_cv(self, frame):
        try:
            self.YOLOCounter += 1
            self.current_frame = frame
            if(self.results is None or self.YOLOCounter%self.YOLOlimit == 0  ):
            # Run inference with the PyTorch model
                thread = threading.Thread(target=self.process_yolo_segment)
                thread.start()
            
            # # Get the annotated image from YOLO
            # annotated_image = results[0].plot()
            if(self.results is None):   
                return frame
            

            # Make a copy of the original frame for custom processing
            custom_frame = frame.copy()
            
            r = self.results[0]
            detected_cats = []
            # Process both boxes and masks
            if hasattr(r, 'boxes') and hasattr(r, 'masks') and r.masks is not None:
                for i, (box, mask) in enumerate(zip(r.boxes, r.masks)):
                    # Box information
                    xyxy = box.xyxy[0].cpu().numpy()      # [x1, y1, x2, y2]
                    conf = box.conf[0].item()             # confidence score
                    cls = int(box.cls[0].item())          # class ID
                    class_name = self.model.names[cls]
                    
                    # Process cats with custom annotations and color analysis
                    if mask is not None and class_name == "cat":
                        # Get the segmentation mask as numpy array
                        segment_mask = mask.data[0].cpu().numpy()
                        
                        # Convert mask to the same size as frame if needed
                        if segment_mask.shape[:2] != frame.shape[:2]:
                            segment_mask = cv2.resize(segment_mask, (frame.shape[1], frame.shape[0]))
                        
                        # Convert to binary mask
                        binary_mask = (segment_mask > 0.5).astype(np.uint8)
                        
                        # Extract color information from the cat segment

                        # Add colored overlay for cat
                        overlay = custom_frame.copy()
                        overlay[binary_mask == 1] = [0, 255, 0]  # Green color (BGR format)
                        custom_frame = cv2.addWeighted(custom_frame, 0.7, overlay, 0.3, 0)
                        
                        # Add contours around the cat
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(custom_frame, contours, -1, (0, 255, 255), 2)  # Yellow contours
                        
                        # Add bounding box
                        x1, y1, x2, y2 = xyxy.astype(int)
                        cv2.rectangle(custom_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle
                        

                        # Get frame dimensions
                        height, width = frame.shape[:2]

                        # Ensure coordinates are within bounds
                        x1 = max(0, int(xyxy[0]))
                        y1 = max(0, int(xyxy[1]))
                        x2 = min(width, int(xyxy[2]))
                        y2 = min(height, int(xyxy[3]))

                        # Crop the frame
                        cropped_frame = frame[y1:y2, x1:x2]

                        focusedout = self.focusedmodel.predict_frame_CV(cropped_frame,True)
                        allcat_dict = focusedout["all_probabilities"]
                        


                        name = focusedout["predicted_cat"]
                        confidence = focusedout["confidence_percent"]

                        if("bunt" in allcat_dict):
                            if(allcat_dict["bunt"]["probability"] > 0.30):
                                name = "bunt"
                                confidence = focusedout["confidence_percent"]
                        
                        detected_cats.insert(name)
                        if(name == "bunt"):
                            self.bunt_counter += 1
                            index = self.bunt_counter/self.bunt_namecountover
                            if(index>=self.bunt_max):
                                self.bunt_counter = 0
                                index = self.bunt_counter/self.bunt_namecountover
                            name = self.bunt_names[int(index)]

                        # self.log_cat_colors_to_csv(dominant_color=dominant_color_bgr,secondary_color=secondary_color,tertiary_color=tertiary_color,color_variation=color_variation,confidence=conf)
                        label = f"Cat: {conf:.2f}"
                        color_label = f"Name: {name}, confidence: {confidence}"
                        
                        # Calculate label sizes
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        color_label_size = cv2.getTextSize(color_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Draw background for labels
                        total_height = label_size[1] + color_label_size[1] + 15
                        max_width = max(label_size[0], color_label_size[0])
                        cv2.rectangle(custom_frame, (x1, y1 - total_height - 5), 
                                    (x1 + max_width + 10, y1), (255, 0, 0), -1)
                        
                        # Draw main label
                        cv2.putText(custom_frame, label, (x1 + 5, y1 - color_label_size[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw color information
                        cv2.putText(custom_frame, color_label, (x1 + 5, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw small color square showing dominant color
                        color_square_size = 20
                        cv2.rectangle(custom_frame, 
                                    (x2 - color_square_size - 5, y1 + 5),
                                    (x2 - 5, y1 + color_square_size + 5),1)
                        cv2.rectangle(custom_frame, 
                                    (x2 - color_square_size - 5, y1 + 5),
                                    (x2 - 5, y1 + color_square_size + 5),
                                    (255, 255, 255), 1)
                        
                       
            
            # Return the custom processed frame instead of the default annotated image
            return custom_frame, detected_cats
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return frame, []  # Return original frame if there's an error

 



    def process_frame(self, image_path, output_name):
        try:
            # Run inference with the PyTorch model
            results = self.model(image_path, task="segment")
            
            # Draw results on image
            annotated_image = results[0].plot()
            cv2.imwrite(f"{output_name}.jpg", annotated_image)
            
            r = results[0]
            
            # Process both boxes and masks
            if hasattr(r, 'boxes') and hasattr(r, 'masks') and r.masks is not None:
                for i, (box, mask) in enumerate(zip(r.boxes, r.masks)):
                    # Box information
                    xyxy = box.xyxy[0].cpu().numpy()      # [x1, y1, x2, y2]
                    conf = box.conf[0].item()             # confidence score
                    cls = int(box.cls[0].item())          # class ID
                    class_name = self.model.names[cls]
                    
                    print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")
                    
                    # Save individual mask if needed
                    if mask is not None and class_name == "cat":
                        # Get the segmentation mask as numpy array
                        segment_mask = mask.data[0].cpu().numpy()
                        
                        # Create a visualization image for the mask
                        # (masks are boolean/float arrays, need conversion for visualization)
                        h, w = segment_mask.shape
                        mask_image = np.zeros((h, w, 3), dtype=np.uint8)
                        mask_image[segment_mask > 0.5] = [0, 255, 0]  # Green color for the mask
                        
                        # Save the mask image
                        mask_filename = f"{output_name}_{class_name}_{i}.jpg"
                        # cv2.imwrite(mask_filename, mask_image)
                        self.process_mask(segment_mask,image_path,mask_filename)
                        print(f"Saved mask for {class_name} as {mask_filename}")
            else:
                print("No objects detected or masks not available.")
        
        except Exception as e:
            print(f"Error processing image: {e}")


class webcam:
    cap = cv2.VideoCapture(0)
    


    def returnFrame(self):
        if(not self.cap.isOpened()):
           return
        ret, frame = self.cap.read()
        return frame


class Engine:

    looping = False
    looparound = True
    cam = webcam()
    processor = Processor()
    waittime = 0.2
    processframe = True

    #loops while   
    def looper(self):
        if(self.looping):
            return
        self.looping = True
        self.looparound = True
        while(self.looparound):
            time.sleep(self.waittime)
            frame = self.cam.returnFrame()
            output = self.processor.process_frame_cv(frame)


            # Display the captured frame
            cv2.imshow('Camera', output)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                self.looping = False
                self.looparound = False
                break


    

# if __name__ == "__main__":
#     MainEngine = Engine()

#     MainEngine.looper()