from ultralytics import YOLO
from src.config import MODEL_PATH, DATA_YAML

model = YOLO(MODEL_PATH)

def evaluate_model():
    results = model.val(data=DATA_YAML)

    metrics = results.box

    summary = (
        f"Precision: {metrics.mp:.4f}\n"
        f"Recall: {metrics.mr:.4f}\n"
        f"mAP@0.5: {metrics.map50:.4f}\n"
        f"mAP@0.5:0.95: {metrics.map:.4f}\n"
        f"f1 score: {2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr + 1e-6):.4f}"
    )

    return summary