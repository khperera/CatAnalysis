import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt

class CatDataset(Dataset):
    """Custom dataset for loading cat images"""
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subdirectories
            transform (callable, optional): Optional transform to be applied on a sample
        
        Expected directory structure:
        root_dir/
            cat1_name/
                image1.jpg
                image2.jpg
                ...
            cat2_name/
                image1.jpg
                image2.jpg
                ...
            cat3_name/
                image1.jpg
                image2.jpg
                ...
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Get all subdirectories (cat names)
        for idx, cat_name in enumerate(sorted(os.listdir(root_dir))):
            cat_dir = os.path.join(root_dir, cat_name)
            if os.path.isdir(cat_dir):
                self.class_names.append(cat_name)
                for img_name in os.listdir(cat_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(cat_dir, img_name))
                        self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CatClassifier:
    """Cat classifier using MobileNet V3 Small"""
    
    def __init__(self, num_cats: int = 3):
        self.num_cats = num_cats
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        
        # Define transforms using MobileNet V3 Small recommended preprocessing
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = weights.transforms()
    
    def build_model(self):
        """Build the model using pre-trained MobileNet V3 Small"""
        # Load pre-trained MobileNet V3 Small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = mobilenet_v3_small(weights=weights)
        
        # Freeze early layers (optional - you can experiment with this)
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Replace the classifier for 3 cats
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, self.num_cats)
        )
        
        self.model = self.model.to(self.device)
        return self.model
    
    def train(self, train_dir: str, val_dir: str = None, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        """Train the model"""
        # Create datasets
        train_dataset = CatDataset(train_dir, transform=self.train_transform)
        self.class_names = train_dataset.class_names
        
        # Split training data for validation if no separate validation directory
        if val_dir is None:
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            # Update val_dataset transform
            val_dataset.dataset.transform = self.val_transform
        else:
            val_dataset = CatDataset(val_dir, transform=self.val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = 100 * correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100 * correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
            
            scheduler.step()
            
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
            print('-' * 50)
        
        # Plot training history
        self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def plot_training_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """Plot training and validation loss and accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """Predict the cat class for a single image"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        predicted_cat = self.class_names[predicted_idx]
        return predicted_cat, confidence
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """Predict cat classes for multiple images"""
        results = []
        for image_path in image_paths:
            pred_cat, confidence = self.predict(image_path)
            results.append((pred_cat, confidence))
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'num_cats': self.num_cats
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.num_cats = checkpoint['num_cats']
        
        # Build model architecture
        self.build_model()
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = CatClassifier(num_cats=3)
    
    # Example training (replace with your actual data paths)
    # Make sure your data is organized as described in CatDataset docstring
    
    # Train the model
    history = classifier.train(
        train_dir="./datasets/catimages/",  # Replace with your data path
        epochs=20,
        batch_size=32,
        lr=0.001
    )
    
    # Save the trained model
    classifier.save_model("cat_classifier_mobilenet_v3.pth")
    
    # Make predictions
    predicted_cat, confidence = classifier.predict("./datasets/catimages/tot/received_566340709819603_jpeg.rf.db8b053d01965656ae34da28f329d30b.jpg")
    print(f"Predicted cat: {predicted_cat} (confidence: {confidence:.2%})")
    
    # Batch predictions
    test_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    results = classifier.predict_batch(test_images)
    for img, (cat, conf) in zip(test_images, results):
        print(f"{img}: {cat} ({conf:.2%})")
    
    
    print("Cat classifier is ready! Please organize your images and update the paths to start training.")