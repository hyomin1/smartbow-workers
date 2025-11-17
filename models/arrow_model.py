from ultralytics import YOLO


class ArrowModel:
    def __init__(self):
        tensorrt_path = "models/arrow_best.onnx"

        print("[ArrowModel] TensorRT 모델 사용")
        self.model = YOLO(tensorrt_path, task="detect")

        self.imgsz = 1280
        self.conf = 0.75
        self.iou = 0.6  #

    def predict(self, frame):
        return self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            max_det=1,
            imgsz=self.imgsz,
            agnostic_nms=True,
            stream=False,
        )[0]
