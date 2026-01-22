from ultralytics import YOLO
from config import DATA_YAML, MODEL_PATH

def main():
    model = YOLO(MODEL_PATH)  # path to your trained model
    results = model.val(data=DATA_YAML)
    print(results)

if __name__ == "__main__":
    main()