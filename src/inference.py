from ultralytics import YOLO
from src.config import MODEL_PATH
import cv2

# Load model ONCE
model = YOLO(MODEL_PATH)

def run_inference(image_path):
    """
    Runs YOLO inference on a single image path.
    Returns: annotated BGR image + raw results
    """
    results = model.predict(image_path)
    img = results[0].orig_img.copy()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"Fall ({conf:.2f})"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return img

