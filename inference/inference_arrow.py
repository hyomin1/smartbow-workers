import cv2, numpy as np, time, msgpack
from weights.arrow_model import ArrowModel
from weights.target_model import TargetModel
from utils.zmq_utils import get_sub_socket, get_pub_socket
from zmq import Poller, POLLIN


def get_bbox_and_tip(result):
    if result is None or len(result.boxes) == 0:
        return None, None, None

    box = result.boxes[0]

    xyxy = box.xyxy.cpu().numpy()[0].tolist()
    conf = float(box.conf.cpu().numpy()[0])

    x1, y1, x2, y2 = xyxy
    tip = [(x1 + x2) / 2, y2]

    return xyxy, tip, conf


class InferenceArrow:
    def __init__(self, cam_id: str, sub_port: int, pub_port: int, gate_port: int):
        self.cam_id = cam_id
        self.sub_port = sub_port
        self.pub_port = pub_port
        self.gate_port = gate_port

        self.arrow_model = ArrowModel()
        self.target_model = TargetModel()

        self.TARGET_UPDATE_INTERVAL = 3600
        self.last_target_update = 0

        self.target = None

        self.fps_count = 0
        self.last_log = time.time()

        self.person_active = False
        self.last_person_seen_ts = 0.0
        self.PERSON_OFF_DELAY = 60.0

    def start(self):
        print(
            f"[InferenceWorker] Start cam={self.cam_id} SUB={self.sub_port} → PUB={self.pub_port}"
        )
        self.run()

    def run(self):
        frame_sub = get_sub_socket(self.sub_port)
        gate_sub = get_sub_socket(self.gate_port)
        pub = get_pub_socket(self.pub_port)

        poller = Poller()
        poller.register(frame_sub, POLLIN)
        poller.register(gate_sub, POLLIN)

        print(f"[InferenceWorker] {self.cam_id} arrow_model loaded")

        while True:
            socks = dict(poller.poll(timeout=100))
            now = time.time()

            if gate_sub in socks:
                gate_msg = gate_sub.recv_json()
                if gate_msg.get('active'):
                    self.person_active = True
                    self.last_person_seen_ts = now

            if self.person_active and now - self.last_person_seen_ts > self.PERSON_OFF_DELAY:
                self.person_active = False

            if frame_sub in socks:
                data = frame_sub.recv()
                msg = msgpack.unpackb(data, raw=False)

                if msg.get('type') != 'frame':
                    continue

                if not self.person_active:
                    continue

                cam_id = msg["cam_id"]
                jpeg = msg["jpeg"]

                t0 = time.time()

                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    print("[WARN] JPEG decode 실패")
                    continue

                result = self.arrow_model.predict(frame)

                bbox, tip, conf = get_bbox_and_tip(result)
                now = time.time()

                h, w = frame.shape[:2]

                if (
                    self.target is None
                    or now - self.last_target_update > self.TARGET_UPDATE_INTERVAL
                ):
                    self.last_target_update = now
                    self.target = self.target_model.predict(frame)
                infer_ms = (time.time() - t0) * 1000
                self.fps_count += 1

                if now - self.last_log >= 1.0:
                    print(f"[{self.cam_id}] FPS={self.fps_count} infer={infer_ms:.1f}ms")
                    self.fps_count = 0
                    self.last_log = now

                if bbox is None:
                    event = {
                        "type": "arrow",
                        "cam_id": cam_id,
                        "bbox": None,
                        "tip": None,
                        "conf": 0.0,
                        "target": self.target,
                        "frame_size": [w, h],
                        "timestamp": time.time(),
                    }
                    pub.send_json(event)
                    continue
                event = {
                    "type": "arrow",
                    "cam_id": cam_id,
                    "bbox": bbox,
                    "tip": tip,
                    "conf": conf if conf else 0.0,
                    "target": self.target,
                    "frame_size": [w, h],
                    "timestamp": time.time(),
                }
                print("화살", event["tip"])

                pub.send_json(event)
