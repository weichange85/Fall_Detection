import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

from src.inference import run_inference

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

        results = run_inference(self.image_path)
        img = results[0].orig_img.copy()

        for (x1, y1, x2, y2), conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"Fall ({conf:.2f})",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        self.display(img)

    def display(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((640, 640))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.tk_img)


if __name__ == "__main__":
    root = tk.Tk()
    FallDetectionApp(root)
    root.mainloop()
