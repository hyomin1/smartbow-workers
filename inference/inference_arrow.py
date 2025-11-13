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


def main():
    SUB_PORTS = [5553]
    PUB_PORT = 5560

    sub = get_sub_socket(SUB_PORTS)
    pub = get_pub_socket(PUB_PORT)

    model = ArrowModel()
    print("[InferenceWorker] Arrow YOLO 모델 로딩 완료")

    while True:

        cam_id, jpeg = sub.recv_multipart()
        cam_id = cam_id.decode()

        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            print("[WARN] JPEG decode 실패")
            continue

        # 2) YOLO 추론
        result = model.predict(frame)

        # 3) bbox + tip 계산
        bbox, tip, conf = get_bbox_and_tip(result)

        if bbox is None:

            event = {
                "type": "arrow",
                "cam_id": cam_id,
                "bbox": None,
                "tip": None,
                "conf": 0.0,
                "timestamp": time.time(),
            }
        else:

            event = {
                "type": "arrow",
                "cam_id": cam_id,
                "bbox": bbox,
                "tip": tip,
                "conf": conf,
                "timestamp": time.time(),
            }
            print("화살 추론", event)

        # 4) FastAPI로 전송
        pub.send_json(event)


if __name__ == "__main__":
    main()
