from ultralytics import YOLO
import cv2
import numpy as np

class Processor:
    def __init__(self):
        # Load the segmentation model directly without exporting to NCNN
        # Using the original PyTorch model which fully supports segmentation
        self.model = YOLO("epoch3000.pt")
        
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


if __name__ == "__main__":
    processor = Processor()
    
    terminal_string = ""
    while terminal_string != "stop":
        terminal_string = input("Enter image path (or 'stop' to exit): ")
        if terminal_string == "stop":
            break
            
        output_name = input("Enter output filename (without extension): ")
        processor.process_frame(terminal_string, output_name)
        print(f"Processed image saved as {output_name}.jpg")