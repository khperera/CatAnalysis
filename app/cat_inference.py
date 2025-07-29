import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import List, Tuple, Dict
import numpy as np
import cv2









class CatInference:
    """Inference class for the trained cat classifier"""
    
    def __init__(self, model_path: str):
        """
        Initialize the inference class with a saved model
        
        Args:
            model_path (str): Path to the saved model file (.pth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.num_cats = 3
        
        # Define the same preprocessing transforms used during training
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load the model
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the saved model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.num_cats = checkpoint['num_cats']
        
        # Rebuild model architecture
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, self.num_cats),
            
        )
        
        # Load weights and set to evaluation mode
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

            # Apply dynamic quantization after loading
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Classes: {self.class_names}")


    def preprocess_frame_CV(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess an OpenCV frame for inference"""
        if frame is None:
            raise ValueError("Frame is None")
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Apply transforms and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def predict_frame_CV(self, frame: np.ndarray, return_probabilities: bool = False) -> Dict:
        """
        Predict the cat class for an OpenCV frame
        
        Args:
            frame (np.ndarray): OpenCV frame (BGR format)
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            Dict: Prediction results containing cat name, confidence, and optionally all probabilities
        """
        # Preprocess frame
        image_tensor = self.preprocess_frame_CV(frame)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get predicted class
            predicted_idx = torch.argmax(probabilities).item()
            predicted_cat = self.class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()
        
        result = {
            'frame_shape': frame.shape,
            'predicted_cat': predicted_cat,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%"
        }
        
        if return_probabilities:
            all_probabilities = {}
            for i, cat_name in enumerate(self.class_names):
                all_probabilities[cat_name] = {
                    'probability': probabilities[i].item(),
                    'percentage': f"{probabilities[i].item() * 100:.1f}%"
                }
            result['all_probabilities'] = all_probabilities
        
        return result



    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for inference"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict_single(self, image_path: str, return_probabilities: bool = False) -> Dict:
        """
        Predict the cat class for a single image
        
        Args:
            image_path (str): Path to the image file
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            Dict: Prediction results containing cat name, confidence, and optionally all probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get predicted class
            predicted_idx = torch.argmax(probabilities).item()
            predicted_cat = self.class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()
        
        result = {
            'image_path': image_path,
            'predicted_cat': predicted_cat,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%"
        }
        
        if return_probabilities:
            all_probabilities = {}
            for i, cat_name in enumerate(self.class_names):
                all_probabilities[cat_name] = {
                    'probability': probabilities[i].item(),
                    'percentage': f"{probabilities[i].item() * 100:.1f}%"
                }
            result['all_probabilities'] = all_probabilities
        
        return result
    
    def predict_batch(self, image_paths: List[str], return_probabilities: bool = False) -> List[Dict]:
        """
        Predict cat classes for multiple images
        
        Args:
            image_paths (List[str]): List of image file paths
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, return_probabilities)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
    
    def predict_directory(self, directory_path: str, return_probabilities: bool = False) -> List[Dict]:
        """
        Predict cat classes for all images in a directory
        
        Args:
            directory_path (str): Path to directory containing images
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            List[Dict]: List of prediction results
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Supported image extensions
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        # Get all image files in directory
        image_paths = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(directory_path, filename))
        
        if not image_paths:
            print(f"No supported image files found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths, return_probabilities)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'num_classes': self.num_cats,
            'class_names': self.class_names,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }

class InteractiveFileSelector:
    """Interactive file selector using tkinter"""
    
    def __init__(self):
        # Initialize tkinter root window (but keep it hidden)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
    def select_model_file(self) -> str:
        """Prompt user to select the trained model file"""
        print("Please select your trained model file (.pth)")
        model_path = filedialog.askopenfilename(
            title="Select Trained Model File",
            filetypes=[
                ("PyTorch Model", "*.pth"),
                ("All Files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if not model_path:
            raise ValueError("No model file selected")
        
        return model_path
    
    def select_inference_mode(self) -> str:
        """Prompt user to select inference mode"""
        modes = {
            "1": "single",
            "2": "batch", 
            "3": "directory"
        }
        
        print("\nSelect inference mode:")
        print("1. Single image prediction")
        print("2. Batch prediction (multiple images)")
        print("3. Directory prediction (all images in folder)")
        
        while True:
            choice = input("Enter your choice (1-3): ").strip()
            if choice in modes:
                return modes[choice]
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    def select_single_image(self) -> str:
        """Prompt user to select a single image"""
        print("\nPlease select an image file for prediction")
        image_path = filedialog.askopenfilename(
            title="Select Image for Prediction",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if not image_path:
            raise ValueError("No image file selected")
        
        return image_path
    
    def select_multiple_images(self) -> List[str]:
        """Prompt user to select multiple images"""
        print("\nPlease select multiple images for batch prediction")
        image_paths = filedialog.askopenfilenames(
            title="Select Images for Batch Prediction",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if not image_paths:
            raise ValueError("No image files selected")
        
        return list(image_paths)
    
    def select_directory(self) -> str:
        """Prompt user to select a directory"""
        print("\nPlease select a directory containing images")
        directory_path = filedialog.askdirectory(
            title="Select Directory with Images",
            initialdir=os.getcwd()
        )
        
        if not directory_path:
            raise ValueError("No directory selected")
        
        return directory_path
    
    def ask_include_probabilities(self) -> bool:
        """Ask user if they want to include all class probabilities"""
        while True:
            choice = input("\nInclude probabilities for all cats? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def ask_save_results(self) -> Tuple[bool, str]:
        """Ask user if they want to save results to file"""
        while True:
            choice = input("\nSave results to JSON file? (y/n): ").strip().lower()
            if choice in ['n', 'no']:
                return False, ""
            elif choice in ['y', 'yes']:
                # Ask for filename
                filename = filedialog.asksaveasfilename(
                    title="Save Results As",
                    defaultextension=".json",
                    filetypes=[
                        ("JSON Files", "*.json"),
                        ("All Files", "*.*")
                    ],
                    initialdir=os.getcwd()
                )
                return bool(filename), filename
            else:
                print("Please enter 'y' for yes or 'n' for no.")

# Utility functions
def print_prediction_result(result: Dict):
    """Pretty print a single prediction result"""
    print(f"\nImage: {os.path.basename(result['image_path'])}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Predicted Cat: {result['predicted_cat']}")
    print(f"Confidence: {result['confidence_percent']}")
    
    if 'all_probabilities' in result:
        print("All probabilities:")
        for cat_name, prob_info in result['all_probabilities'].items():
            print(f"  {cat_name}: {prob_info['percentage']}")

def save_results_to_json(results: List[Dict], output_path: str):
    """Save prediction results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def print_summary(results: List[Dict]):
    """Print a summary of batch/directory results"""
    if not results:
        print("No results to summarize")
        return
    
    print(f"\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Total images processed: {len(results)}")
    
    # Count predictions by cat
    cat_counts = {}
    error_count = 0
    
    for result in results:
        if 'error' in result:
            error_count += 1
        elif 'predicted_cat' in result:
            cat = result['predicted_cat']
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    if cat_counts:
        print("\nPredictions by cat:")
        for cat, count in sorted(cat_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {cat}: {count} images ({percentage:.1f}%)")
    
    if error_count > 0:
        print(f"\nErrors encountered: {error_count}")

# Main interactive script
def main():
    """Main interactive inference script"""
    print("="*60)
    print("CAT CLASSIFIER INFERENCE - INTERACTIVE MODE")
    print("="*60)
    
    try:
        # Initialize file selector
        file_selector = InteractiveFileSelector()
        
        # Step 1: Select model file
        print("\nStep 1: Select your trained model")
        model_path = file_selector.select_model_file()
        print(f"Selected model: {os.path.basename(model_path)}")
        
        # Step 2: Load model
        print("\nStep 2: Loading model...")
        cat_inference = CatInference(model_path)
        
        # Print model info
        model_info = cat_inference.get_model_info()
        print(f"\nModel Information:")
        print(f"  Classes: {model_info['class_names']}")
        print(f"  Device: {model_info['device']}")
        print(f"  Parameters: {model_info['model_parameters']:,}")
        
        # Step 3: Select inference mode
        print("\nStep 3: Select inference mode")
        mode = file_selector.select_inference_mode()
        
        # Step 4: Ask about probabilities
        include_probabilities = file_selector.ask_include_probabilities()
        
        # Step 5: Perform inference based on mode
        results = []
        
        if mode == "single":
            image_path = file_selector.select_single_image()
            print(f"\nProcessing: {os.path.basename(image_path)}")
            result = cat_inference.predict_single(image_path, include_probabilities)
            print_prediction_result(result)
            results = [result]
            
        elif mode == "batch":
            image_paths = file_selector.select_multiple_images()
            print(f"\nProcessing {len(image_paths)} images...")
            results = cat_inference.predict_batch(image_paths, include_probabilities)
            
            # Print individual results
            for result in results:
                print_prediction_result(result)
            
            # Print summary
            print_summary(results)
            
        elif mode == "directory":
            directory_path = file_selector.select_directory()
            print(f"\nProcessing directory: {os.path.basename(directory_path)}")
            results = cat_inference.predict_directory(directory_path, include_probabilities)
            
            # Print individual results
            for result in results:
                print_prediction_result(result)
            
            # Print summary
            print_summary(results)
        
        # Step 6: Ask about saving results
        if results:
            save_results, save_path = file_selector.ask_save_results()
            if save_results and save_path:
                save_results_to_json(results, save_path)
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except ValueError as e:
        print(f"\nOperation cancelled: {e}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your model file and input images.")
    
    finally:
        # Clean up tkinter
        try:
            file_selector.root.quit()
            file_selector.root.destroy()
        except:
            pass

if __name__ == "__main__":
    main()