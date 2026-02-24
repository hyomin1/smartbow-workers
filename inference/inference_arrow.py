import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import zmq
from roboflow import Roboflow

from config import ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_ID, ROBOFLOW_WORKSPACE
from utils.frame_shm import FrameBuffer
from utils.zmq_utils import get_pub_socket, get_rep_socket, get_sub_socket
from weights.arrow_model import ArrowModel
from weights.splash_model import SplashModel
from weights.target_model import TargetModel
from workers.base import BaseWorker


class InferenceArrow(BaseWorker):
    def __init__(
        self,
        cam_id: str,
        pub_port: str,
        gate_port: str,
        target_port: str,
        zone_port: str,
        shape,
    ):
        super().__init__(f"Camera-{cam_id}")
        self.cam_id = cam_id
        self.pub_port = pub_port  # Fast API용 추론 결과 전송
        self.gate_port = gate_port  # 사람 추론 워커에서 사람 추론 결과 송신
        self.target_port = target_port
        self.zone_port = zone_port

        self.shm_name = f"shm_{cam_id}"
        self.frame_buffer = FrameBuffer(name=self.shm_name, shape=shape, create=False)

        self.arrow_model = ArrowModel()
        self.target_model = TargetModel()
        self.splash_model = SplashModel()

        self.prev_frame = None

        self.frame_seq = 0

        self.TARGET_UPDATE_INTERVAL = 3600
        self.last_target_update = 0

        self.target = None
        self.target_rep = None

        self.last_sent_tip = None  # 같은 위치 추론한 화살 안보내기 위한용도

        self.fps_count = 0
        self.last_log = time.time()

        self.person_active = False
        self.last_person_seen_ts = 0.0
        self.PERSON_OFF_DELAY = 40.0

        self.case_b_buffer = deque(maxlen=4)  # 오탐 영역 찾는 버퍼

        self.batch_id = None

        self.kernel = np.ones((3, 3), np.uint8)

        try:
            self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            self.rf_project = self.rf.workspace(ROBOFLOW_WORKSPACE).project(
                ROBOFLOW_PROJECT_ID
            )
            self.upload_enabled = True
        except Exception as e:
            print("roboflow error", e)
            self.upload_enabled = False

    def _async_upload(self, frame, conf, ts, bbox=None, frame_seq=None, batch_id=None):
        conf_int = int(conf * 100)
        date_obj = datetime.fromtimestamp(ts)
        date_str = date_obj.strftime("%Y%m%d_%H%M%S")

        base_name = f"{date_str}_f{frame_seq}_conf{conf_int}"
        raw_filename = f"{base_name}.jpg"
        debug_filename = f"debug_{base_name}.jpg"

        batch_folder_name = (
            f"{date_obj.strftime('%Y%m%d')}_Batch_{batch_id}"
            if batch_id
            else "General_Upload"
        )
        try:
            cv2.imwrite(raw_filename, frame)
            self.rf_project.upload(
                raw_filename, num_retry_uploads=3, batch_name=batch_folder_name
            )
            if bbox is not None:
                debug_frame = frame.copy()
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                debug_root = "debug_data"
                target_debug_dir = os.path.join(debug_root, batch_folder_name)
                if not os.path.exists(target_debug_dir):
                    os.makedirs(target_debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(target_debug_dir, debug_filename), debug_frame)
        except Exception as e:
            print("roboflow error", e)
        finally:
            if os.path.exists(raw_filename):
                os.remove(raw_filename)

    def init_target(self):
        print(f"[{self.cam_id}] 과녁 영역 초기화")
        while self.running:
            frame = self.frame_buffer.read()
            if frame is None:
                time.sleep(0.1)
                continue

            target = self.target_model.predict(frame)
            if target:
                self.target = target
                print(f"[{self.cam_id}] Target initialized")
                return

    def handle_gate_msg(self, msg):
        now = time.time()
        if msg.get("active"):
            if not self.person_active:
                self.batch_id = self.current_batch_id = datetime.fromtimestamp(
                    now
                ).strftime("%H%M%S")
            self.person_active = True
            self.last_person_seen_ts = now

    def process_frame(self, frame):
        self.frame_seq += 1

        result = self.arrow_model.predict(frame)
        if (
            result is not None
            and result.boxes is not None
            and result.keypoints is not None
            and len(result.boxes) > 0
            and len(result.keypoints) > 0
        ):
            if len(result.boxes) > 0 and len(result.keypoints) > 0:
                box = result.boxes.xyxy[0].cpu().numpy()
                bbox_conf = round(result.boxes.conf[0].item(), 2)

                # if 0.4 <= bbox_conf < 0.59 and self.upload_enabled:
                #     ts = time.time()
                #     threading.Thread(
                #         target=self._async_upload,
                #         args=(
                #             frame.copy(),
                #             bbox_conf,
                #             ts,
                #             box.tolist(),
                #             self.frame_seq,
                #             self.current_batch_id,
                #         ),
                #         daemon=True,
                #     ).start()
                #     return None

                if bbox_conf >= 0.6:
                    if self.prev_frame is not None:
                        x1, y1, x2, y2 = map(int, box)
                        curr_crop = cv2.cvtColor(
                            frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                        )
                        prev_crop = cv2.cvtColor(
                            self.prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                        )
                        diff = cv2.absdiff(curr_crop, prev_crop)
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        mask = cv2.erode(thresh, self.kernel, iterations=1)
                        mask = cv2.dilate(mask, self.kernel, iterations=1)
                        motion_count = np.count_nonzero(mask)
                        motion_ratio = motion_count / mask.size
                        # print(f"motion_ratio: {motion_ratio:.4f}")

                        if motion_ratio < 0.001:  # 0.1% 미만 움직임은 정지 상태로 간주
                            # print(f"motion_ratio: {motion_ratio:.6f}")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                frame,
                                f"Ignored: {motion_ratio:.4f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                            self.prev_frame = frame.copy()
                            return None

                        # if motion_score < 10.0:  # 임계값, 튜닝 필요
                        #     # debug_path = f"debug_diff/diff_f{self.frame_seq}_{int(motion_score * 100)}.jpg"
                        #     # os.makedirs("debug_diff", exist_ok=True)
                        #     # cv2.imwrite(debug_path, diff)
                        #     self.prev_frame = frame.copy()
                        #     return None

                    kpts = result.keypoints.data[0].cpu().numpy()

                    tip_x, tip_y, tip_conf = kpts[0]
                    tail_x, tail_y, tail_conf = kpts[1]

                    return {
                        "cam_id": self.cam_id,
                        "frame_seq": self.frame_seq,
                        "bbox": box.tolist(),
                        "bbox_conf": float(bbox_conf),
                        "tip": [int(tip_x), int(tip_y)],
                        "tail": [int(tail_x), int(tail_y)],
                        "timestamp": time.time(),
                    }
        self.prev_frame = frame.copy()
        return None

    def is_inside_target(self, cx, cy):
        if self.target is None:
            return False
        target = np.array(self.target, dtype=np.int32)
        return cv2.pointPolygonTest(target, (cx, cy), False) >= 0

    def handle_case_b(self, bbox, zone_pub):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if self.is_inside_target(cx, cy):
            self.case_b_buffer.clear()
            return

        self.case_b_buffer.append((cy, bbox))

        if len(self.case_b_buffer) < 4:
            return

        ys = [p[0] for p in self.case_b_buffer]
        if max(ys) - min(ys) < 3:
            print("오탐 영역 발견")
            zone_pub.send_json(
                {
                    "type": "static_zone",
                    "cam_id": self.cam_id,
                    "bbox": bbox,
                    "pad_ratio": 0.3,
                    "timestamp": time.time(),
                }
            )

            self.case_b_buffer.clear()

    def start(self):
        print(f"[InferenceWorker] Start cam={self.cam_id}")
        self.target_rep = get_rep_socket(self.target_port)
        self.init_target()

        self.run()

    def run(self):
        gate_sub = get_sub_socket(self.gate_port)
        pub = get_pub_socket(self.pub_port)
        zone_pub = get_pub_socket(self.zone_port)

        print(f"[InferenceWorker] {self.cam_id} arrow_model loaded")

        while self.running:
            now = time.time()
            try:
                assert self.target_rep
                msg = self.target_rep.recv_json(flags=zmq.NOBLOCK)
                if msg.get("type") == "get_target":
                    h, w = self.frame_buffer.shape[:2]
                    self.target_rep.send_json(
                        {
                            "type": "target_init",
                            "cam_id": self.cam_id,
                            "target": self.target,
                            "frame_size": [w, h],
                            "timestamp": time.time(),
                        }
                    )
            except zmq.Again:
                pass
            try:
                msg = gate_sub.recv_json(flags=zmq.NOBLOCK)
                self.handle_gate_msg(msg)
            except zmq.Again:
                pass

            if (
                self.person_active
                and now - self.last_person_seen_ts > self.PERSON_OFF_DELAY
            ):
                self.sleep(0.01)
                self.person_active = False
                self.case_b_buffer.clear()
                self.last_sent_tip = None
                zone_pub.send_json(
                    {
                        "type": "reset_zone",
                        "cam_id": self.cam_id,
                        "timestamp": time.time(),
                    }
                )

            if not self.person_active:
                continue

            frame = self.frame_buffer.read()
            if frame is None:
                self.sleep(0.01)
                continue

            t_start = time.time()
            analysis = self.process_frame(frame)
            splash_bbox, splash_conf = self.splash_model.predict(frame)
            if splash_bbox is not None:
                print(splash_bbox, splash_conf)
                pub.send_json(
                    {"type": "splash", "splash_bbox": splash_bbox, "conf": splash_conf}
                )

            if analysis is not None:
                tip = (analysis["tip"][0], analysis["tip"][1])

                if tip == self.last_sent_tip:
                    pass
                else:
                    self.last_sent_tip = tip
                    pub.send_json({"type": "arrow", **analysis})

            # if analysis and analysis["case"] != "NONE":
            #     if analysis["case"] == "B":
            #         self.handle_case_b(analysis["bbox"], zone_pub)
            #     else:
            #         self.case_b_buffer.clear()
            #     pub.send_json({"type": "arrow", **analysis})

            self.fps_count += 1
            if now - self.last_log >= 1.0:
                dur = (time.time() - t_start) * 1000
                print(f"[{self.cam_id}] FPS={self.fps_count} last_infer={dur:.1f}ms")
                self.fps_count = 0
                self.last_log = now
