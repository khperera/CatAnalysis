import tkinter as tk
from tkinter import ttk, font
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import app.AppV3
import app.StateTracker as State


class OpenCVGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV GUI")
        self.root.geometry("1000x600")

        self.engine = app.AppV3.Processor()

        
        # Application state
        self.is_open = False
        self.is_ai_on = False
        self.settings = {
            "threshold": 50,
            "fps": 30,
            "sensitivity": 5
        }
        
        # Setup UI fonts
        self.button_font = font.Font(family="Helvetica", size=14, weight="bold")
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create settings frame (initially hidden)
        self.settings_frame = tk.Frame(self.root)
        
        # Setup main UI
        self.setup_main_ui()
        
        # Setup settings UI
        self.setup_settings_ui()
        
        
        # Initialize video capture
        self.cap = None
        self.video_thread = None
        self.running = False
        self.Controller = State.StateController()

        # Start video capture
        self.start_video()

    def setup_main_ui(self):
        # Create a frame for the camera view
        self.camera_frame = tk.Frame(self.main_frame, bg="black")
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a label for displaying the camera feed
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for buttons
        self.button_frame = tk.Frame(self.main_frame, width=200)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Create buttons with good size for touchscreen
        self.manual_button = tk.Button(
            self.button_frame, 
            text="Hatch Open", 
            command=self.toggle_manual,
            font=self.button_font,
            bg="#4CAF50",
            fg="white",
            height=3,
            relief=tk.RAISED,
            bd=3
        )
        self.manual_button.pack(fill=tk.X, pady=10)
        
        self.ai_button = tk.Button(
            self.button_frame, 
            text="Turn On AI", 
            command=self.toggle_ai,
            font=self.button_font,
            bg="#2196F3",
            fg="white",
            height=3,
            relief=tk.RAISED,
            bd=3
        )
        self.ai_button.pack(fill=tk.X, pady=10)
        
        self.settings_button = tk.Button(
            self.button_frame, 
            text="Settings", 
            command=self.show_settings,
            font=self.button_font,
            bg="#FFC107",
            fg="black",
            height=3,
            relief=tk.RAISED,
            bd=3
        )
        self.settings_button.pack(fill=tk.X, pady=10)

    def setup_settings_ui(self):
        # Title
        settings_title = tk.Label(
            self.settings_frame, 
            text="Settings", 
            font=font.Font(family="Helvetica", size=18, weight="bold"),
            pady=15
        )
        settings_title.pack()
        
        # Create a frame for settings controls
        controls_frame = tk.Frame(self.settings_frame, padx=20, pady=10)
        controls_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create settings input fields
        settings_items = [
            ("Threshold:", "threshold"),
            ("FPS:", "fps"),
            ("Sensitivity:", "sensitivity")
        ]
        
        self.setting_entries = {}
        
        for i, (label_text, setting_key) in enumerate(settings_items):
            # Container frame for each setting
            setting_container = tk.Frame(controls_frame)
            setting_container.pack(fill=tk.X, pady=10)
            
            # Label
            label = tk.Label(
                setting_container, 
                text=label_text,
                font=font.Font(family="Helvetica", size=14),
                width=12,
                anchor="w"
            )
            label.pack(side=tk.LEFT)
            
            # Current value label
            current_value = tk.Label(
                setting_container,
                text=f"Current: {self.settings[setting_key]}",
                font=font.Font(family="Helvetica", size=14),
                width=15,
                anchor="w"
            )
            current_value.pack(side=tk.LEFT, padx=(0, 10))
            
            # Entry field
            entry = tk.Entry(
                setting_container,
                font=font.Font(family="Helvetica", size=14),
                width=10
            )
            entry.insert(0, str(self.settings[setting_key]))
            entry.pack(side=tk.LEFT)
            
            self.setting_entries[setting_key] = {
                "entry": entry,
                "current_label": current_value
            }
        
        # Buttons container
        buttons_frame = tk.Frame(self.settings_frame)
        buttons_frame.pack(pady=20)
        
        # Save button
        save_button = tk.Button(
            buttons_frame,
            text="Save",
            command=self.save_settings,
            font=self.button_font,
            bg="#4CAF50",
            fg="white",
            width=10,
            height=2
        )
        save_button.pack(side=tk.LEFT, padx=10)
        
        # Back button
        back_button = tk.Button(
            buttons_frame,
            text="Back",
            command=self.show_main,
            font=self.button_font,
            bg="#F44336",
            fg="white",
            width=10,
            height=2
        )
        back_button.pack(side=tk.LEFT, padx=10)

    def toggle_manual(self):
        self.is_open = not self.is_open


       


        if self.is_open:
            
            hatchthread = threading.Thread(target=self.Controller.CloseValve)
            hatchthread.daemon = True
            hatchthread.start()
            self.manual_button.config(text="Close Hatch ", bg="#F44336")
        else:
            hatchthread2 = threading.Thread(target=self.Controller.OpenValve)
            hatchthread2.daemon = True
            hatchthread2.start()

            self.manual_button.config(text="Open Hatch", bg="#4CAF50")

    def toggle_ai(self):
        self.is_ai_on = not self.is_ai_on
        
        if self.is_ai_on:
            self.ai_button.config(text="Turn Off AI", bg="#9C27B0")
        else:
            self.ai_button.config(text="Turn On AI", bg="#2196F3")

    def show_settings(self):
        self.main_frame.pack_forget()
        self.settings_frame.pack(fill=tk.BOTH, expand=True)

    def show_main(self):
        self.settings_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def save_settings(self):
        # Update settings with values from entry fields
        for key, components in self.setting_entries.items():
            try:
                value = components["entry"].get()
                
                # Convert to appropriate type (int or float)
                if key in ["threshold", "fps"]:
                    value = int(value)
                else:
                    value = float(value)
                    
                self.settings[key] = value
                components["current_label"].config(text=f"Current: {value}")
                
            except ValueError:
                # If conversion fails, keep the old value
                components["entry"].delete(0, tk.END)
                components["entry"].insert(0, str(self.settings[key]))
        
        # Return to main screen
        self.show_main()

    def start_video(self):
        """Start the video capture thread"""
        self.cap = cv2.VideoCapture(0)  # Use default camera
        if not self.cap.isOpened():
            # If camera not available, create a dummy feed
            print("Warning: Camera not available. Using dummy feed.")
        
        self.running = True
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()

    def process_ai_frame(self):
        pass

    def update_frame(self):
        """Update the frame continuously in a separate thread"""
        while self.running:
            if self.cap and self.cap.isOpened():
                detectedcats = []
                ret, frame = self.cap.read()
                if ret:
                    # Process the frame if AI is on
                    if self.is_ai_on:
                        # frame = self.process_frame_ai(frame)
                        frame, detectedcats = self.engine.process_frame_cv(frame)

                    self.Controller
                    # Add visual indicator for manual open/close state
                    status_text = "Hatch: OPEN" if self.is_open else "Hatch: CLOSED"
                    ai_status = "AI: ON" if self.is_ai_on else "AI: OFF"
                    
                    # Draw status on frame
                    cv2.putText(
                        frame, 
                        status_text, 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0) if self.is_open else (0, 0, 255), 
                        2
                    )
                    
                    cv2.putText(
                        frame, 
                        ai_status, 
                        (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 0, 255) if self.is_ai_on else (200, 200, 200), 
                        2
                    )
                    
                    # Convert to RGB for tkinter
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage and update the label
                    self.update_photo(rgb_frame)
            else:
                # Create a dummy frame if camera not available
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    dummy_frame, 
                    "Camera not available", 
                    (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )
                self.update_photo(dummy_frame)
            
            # Control the update rate
            time.sleep(1 / self.settings["fps"])

    def update_photo(self, frame):
        """Convert frame to PhotoImage and update GUI label"""
        h, w = frame.shape[:2]
        
        # Get current display size
        display_w = self.camera_label.winfo_width()
        display_h = self.camera_label.winfo_height()
        
        # Skip if display size is not yet available
        if display_w <= 1 or display_h <= 1:
            image = Image.fromarray(frame)
        else:
            # Resize to fit display while maintaining aspect ratio
            ratio = min(display_w/w, display_h/h)
            new_size = (int(w*ratio), int(h*ratio))
            image = Image.fromarray(frame).resize(new_size)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=image)
        
        # Update label
        self.camera_label.config(image=photo)
        self.camera_label.image = photo  # Keep a reference to prevent garbage collection

    def process_frame_ai(self, frame):
        """Apply some simple processing to simulate AI"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold based on settings
        _, thresh = cv2.threshold(gray, self.settings["threshold"], 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # Add a text indicator that AI processing is applied
        cv2.putText(
            frame, 
            f"AI Processing: t={self.settings['threshold']}, s={self.settings['sensitivity']}", 
            (10, frame.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 0), 
            1
        )
        
        return frame

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()

# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = OpenCVGUI(root)
    
    # Ensure cleanup when window closes
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    
    root.mainloop()