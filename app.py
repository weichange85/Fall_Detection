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

    def display(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((640, 640))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.tk_img)


if __name__ == "__main__":
    root = tk.Tk()
    FallDetectionApp(root)
    root.mainloop()
