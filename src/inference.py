from ultralytics import YOLO
from config import MODEL_PATH, INFERENCE_IMG_PATH
import matplotlib.pyplot as plt
import cv2

def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(INFERENCE_IMG_PATH)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    ori_img = results[0].orig_img   # BGR image (numpy array)

    # ---- Draw bounding boxes ----
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        label = f"Fall ({conf:.2f})"

        cv2.rectangle(
            ori_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

        cv2.putText(
            ori_img,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # ---- Convert BGR -> RGB for matplotlib ----
    ori_img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # ---- Display ----
    plt.figure(figsize=(6, 6))
    plt.imshow(ori_img_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
