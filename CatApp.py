import tkinter as tk
from tkinter import font
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import app.AppV3 as AppV3


class CatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Detector")
        self.root.configure(bg="#1a1a2e")

        self.engine = AppV3.Processor()
        self.is_ai_on = False
        self.running = False
        self.cap = None
        self.video_thread = None

        self._build_ui()
        self._start_video()

    def _build_ui(self):
        btn_font = font.Font(family="Helvetica", size=13, weight="bold")

        # Top bar with title and AI toggle
        top_bar = tk.Frame(self.root, bg="#16213e", pady=6)
        top_bar.pack(fill=tk.X)

        tk.Label(
            top_bar, text="Cat Detector", bg="#16213e", fg="white",
            font=font.Font(family="Helvetica", size=16, weight="bold")
        ).pack(side=tk.LEFT, padx=14)

        self.ai_button = tk.Button(
            top_bar, text="Turn On AI", command=self._toggle_ai,
            font=btn_font, bg="#2196F3", fg="white",
            relief=tk.FLAT, padx=16, pady=4, cursor="hand2"
        )
        self.ai_button.pack(side=tk.RIGHT, padx=14)

        # Camera display (fills the rest of the window)
        self.camera_label = tk.Label(self.root, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

    def _toggle_ai(self):
        self.is_ai_on = not self.is_ai_on
        if self.is_ai_on:
            self.ai_button.config(text="Turn Off AI", bg="#9C27B0")
        else:
            self.ai_button.config(text="Turn On AI", bg="#2196F3")

    def _start_video(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Camera not available.")
        self.running = True
        self.video_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.video_thread.start()

    def _update_loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    if self.is_ai_on:
                        result = self.engine.process_frame_cv(frame)
                        # process_frame_cv returns (frame, cats) or just frame if no results yet
                        if isinstance(result, tuple):
                            frame = result[0]
                        else:
                            frame = result

                    # AI status overlay
                    ai_color = (255, 80, 255) if self.is_ai_on else (180, 180, 180)
                    ai_text = "AI: ON" if self.is_ai_on else "AI: OFF"
                    cv2.putText(frame, ai_text, (10, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, ai_color, 2)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._display(rgb)
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera not available", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self._display(blank)

            time.sleep(1 / 30)

    def _display(self, rgb_frame):
        h, w = rgb_frame.shape[:2]
        label_w = self.camera_label.winfo_width()
        label_h = self.camera_label.winfo_height()

        if label_w > 1 and label_h > 1:
            ratio = min(label_w / w, label_h / h)
            new_size = (int(w * ratio), int(h * ratio))
            image = Image.fromarray(rgb_frame).resize(new_size, Image.BILINEAR)
        else:
            image = Image.fromarray(rgb_frame)

        photo = ImageTk.PhotoImage(image=image)
        self.camera_label.config(image=photo)
        self.camera_label.image = photo

    def cleanup(self):
        self.running = False
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x620")
    app = CatApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    root.mainloop()
