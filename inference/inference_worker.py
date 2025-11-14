import cv2
import numpy as np
import time
from models.arrow_model import ArrowModel
from utils.zmq_utils import get_sub_socket, get_pub_socket


def get_bbox_and_tip(result):
    if result is None or len(result.boxes) == 0:
        return None, None, None

    box = result.boxes[0]

    xyxy = box.xyxy.cpu().numpy()[0].tolist()
    conf = float(box.conf.cpu().numpy()[0])

    x1, y1, x2, y2 = xyxy
    tip = [(x1 + x2) / 2, y2]

    return xyxy, tip, conf


class InferenceWorker:
    def __init__(self, cam_id: str, sub_port: int, pub_port: int):
        self.cam_id = cam_id
        self.sub_port = sub_port
        self.pub_port = pub_port
        self.model = ArrowModel()

    def start(self):
        print(
            f"[InferenceWorker] Start cam={self.cam_id} SUB={self.sub_port} → PUB={self.pub_port}"
        )
        self.run()

    def run(self):
        sub = get_sub_socket(self.sub_port)
        pub = get_pub_socket(self.pub_port)

        print(f"[InferenceWorker] {self.cam_id} model loaded")

        while True:
            cam_id, jpeg = sub.recv_multipart()
            cam_id = cam_id.decode()

            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                print("[WARN] JPEG decode 실패")
                continue

            # YOLO inference
            result = self.model.predict(frame)

            bbox, tip, conf = get_bbox_and_tip(result)

            event = {
                "type": "arrow",
                "cam_id": cam_id,
                "bbox": bbox,
                "tip": tip,
                "conf": conf if conf else 0.0,
                "timestamp": time.time(),
            }

            pub.send_json(event)
