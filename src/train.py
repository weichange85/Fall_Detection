# src/train.py

from ultralytics import YOLO
from config import MODEL_NAME, DATA_YAML, EPOCHS, IMGSZ, BATCH_SIZE

def main():
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE
    )

if __name__ == "__main__":
    main()
