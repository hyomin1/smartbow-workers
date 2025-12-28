import time

import cv2
import numpy as np
import zmq

from utils.frame_shm import FrameBuffer
from utils.zmq_utils import get_pub_socket, get_sub_socket
from weights.arrow_model import ArrowModel
from weights.target_model import TargetModel


def get_bbox(result):
    if result is None or len(result.boxes) == 0:
        return None

    box = result.boxes[0]

    xyxy = box.xyxy.cpu().numpy()[0].tolist()

    return xyxy


class InferenceArrow:
    def __init__(self, cam_id: str, pub_port: str, gate_port: str, shape):
        self.cam_id = cam_id
        self.pub_port = pub_port  # Fast API용 추론 결과 전송
        self.gate_port = gate_port  # 사람 추론 워커에서 사람 추론 결과 송신

        self.shm_name = f"shm_{cam_id}"
        self.frame_buffer = FrameBuffer(name=self.shm_name, shape=shape, create=False)

        self.arrow_model = ArrowModel()
        self.target_model = TargetModel()

        self.prev_frame = None
        self.frame_seq = 0

        self.TARGET_UPDATE_INTERVAL = 3600
        self.last_target_update = 0

        self.target = None

        self.fps_count = 0
        self.last_log = time.time()

        self.person_active = False
        self.last_person_seen_ts = 0.0
        self.PERSON_OFF_DELAY = 60.0

    def handle_gate_msg(self, msg):
        now = time.time()
        if msg.get("active"):
            self.person_active = True
            self.last_person_seen_ts = now

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        self.frame_seq += 1

        motion_line = self.get_motion_lines(frame)
        result = self.arrow_model.predict(frame)
        bbox = get_bbox(result)

        case = "NONE"
        if bbox and motion_line:
            case = "A"
        elif bbox:
            case = "B"
        elif motion_line:
            case = "C"

        now = time.time()
        if (
            self.target is None
            or now - self.last_target_update > self.TARGET_UPDATE_INTERVAL
        ):
            self.last_target_update = now
            self.target = self.target_model.predict(frame)

        return {
            "cam_id": self.cam_id,
            "frame_seq": self.frame_seq,
            "case": case,
            "bbox": bbox,
            "motion_line": motion_line,
            "target": self.target,
            "frame_size": [w, h],
            "timestamp": time.time(),
        }

    def log_status(self, res):
        case = "NONE"
        if res["bbox"] and res["motion_line"]:
            case = "Case A"
        elif res["bbox"]:
            case = "Case B"
        elif res["motion_line"]:
            case = "Case C"

        if case != "NONE":
            print(
                f"{self.cam_id} [{res['frame_seq']}] {case} | BBOX: {res['bbox']} | LINE: {res['motion_line']}"
            )

    def get_motion_lines(self, frame):
        if self.frame_seq % 2 != 0:
            return None
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = curr_gray
            return None
        diff = cv2.absdiff(self.prev_frame, curr_gray)

        _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=60,
            maxLineGap=20,
        )
        self.prev_frame = curr_gray

        if lines is not None:
            return lines[0][0].tolist()
        return None

    def start(self):
        print(f"[InferenceWorker] Start cam={self.cam_id}")
        self.run()

    def run(self):
        gate_sub = get_sub_socket(self.gate_port)
        pub = get_pub_socket(self.pub_port)

        print(f"[InferenceWorker] {self.cam_id} arrow_model loaded")

        while True:
            now = time.time()
            try:
                msg = gate_sub.recv_json(flags=zmq.NOBLOCK)
                self.handle_gate_msg(msg)
            except zmq.Again:
                pass

            if (
                self.person_active
                and now - self.last_person_seen_ts > self.PERSON_OFF_DELAY
            ):
                self.person_active = False

            if not self.person_active:
                continue

            frame = self.frame_buffer.read()
            t_start = time.time()
            analysis = self.process_frame(frame)

            if analysis and analysis["case"] != "NONE":
                pub.send_json({"type": "arrow", **analysis})
                self.log_status(analysis)

            self.fps_count += 1
            if now - self.last_log >= 1.0:
                dur = (time.time() - t_start) * 1000
                print(f"[{self.cam_id}] FPS={self.fps_count} last_infer={dur:.1f}ms")
                self.fps_count = 0
                self.last_log = now
