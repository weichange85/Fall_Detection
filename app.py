import subprocess
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

from src.inference import run_inference
from src.eval import evaluate_model

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall Detection")
        self.root.geometry("800x700")
        self.eval_frame = tk.Frame(root)

        self.image_path = None

        tk.Button(root, text="Load Image", command=self.load_image).pack(pady=5)
        tk.Button(root, text="Run Inference", command=self.run_inference).pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        tk.Label(root, text="").pack(pady=10)  # spacer
        
        tk.Button(root, text="Evaluate Model", command=self.run_evaluation).pack(pady=5)

        self.metrics_text = tk.Text(root, height=6, width=60)
        self.metrics_text.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not self.image_path:
            return

        img = cv2.imread(self.image_path)
        self.display(img)

    def run_inference(self):
        if not self.image_path:
            return
        
        img = run_inference(self.image_path)

        self.display(img)

    def run_evaluation(self):
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "Running evaluation...\n")
        self.root.update()

        summary = evaluate_model()

        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, summary)
        self.show_eval_buttons()

    def display(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((640, 640))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.tk_img)

    def show_eval_buttons(self):
        self.eval_frame.pack(pady=10)
        tk.Button(self.eval_frame, 
                  text="Show F1 confidence Curve", 
                  command=lambda: self.open_image("runs/detect/val/BoxF1_curve.png")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Precision-Recall Curve", 
                  command=lambda: self.open_image("runs/detect/val/BoxPR_curve.png")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Precision confidence Curve", 
                  command=lambda: self.open_image("runs/detect/val/BoxP_curve.png")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Recall confidence Curve", 
                  command=lambda: self.open_image("runs/detect/val/BoxR_curve.png")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Confusion matrix (Normalized)", 
                  command=lambda: self.open_image("runs/detect/val/confusion_matrix_normalized.png")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Confision matrix", 
                  command=lambda: self.open_image("runs/detect/val/confusion_matrix.png")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Example Batch 1", 
                  command=lambda: self.open_image("runs/detect/val/val_batch0_pred.jpg")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Example Batch 2", 
                  command=lambda: self.open_image("runs/detect/val/val_batch1_pred.jpg")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.eval_frame, 
                  text="Show Example Batch 3", 
                  command=lambda: self.open_image("runs/detect/val/val_batch2_pred.jpg")).pack(side=tk.LEFT, padx=5)
        
    def open_image(self, path):
        full_path = os.path.abspath(path)

        if os.path.exists(full_path):
            os.startfile(full_path)
        else:
            print("File not found:", full_path)

            
if __name__ == "__main__":
    root = tk.Tk()
    FallDetectionApp(root)
    root.mainloop()
