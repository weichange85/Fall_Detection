from ultralytics import YOLO
from config import MODEL_PATH, INFERENCE_IMG_PATH
import cv2

def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(INFERENCE_IMG_PATH)

    # Show result
    results.show()

if __name__ == "__main__":
    main()