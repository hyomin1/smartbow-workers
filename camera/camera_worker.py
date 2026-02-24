import time
import traceback

import cv2
import zmq

from utils.frame_shm import FrameBuffer
from utils.zmq_utils import get_sub_socket
from workers.base import BaseWorker


class CameraWorker(BaseWorker):
    RETRY_OFFLINE_SEC = 8.0
    RETRY_DISCONNECT_SEC = 0.5

    def __init__(
        self,
        cam_id: str,
        source: str,
        shape=(1080, 1920, 3),
        crop=None,
        zone_port: str | None = None,
    ):
        super().__init__(f"Camera-{cam_id}")

        self.cam_id = cam_id
        self.source = source
        self.zone_port = zone_port
        self.crop = crop

        self.person_roi = (0, 320, 800, 820)

        self.shm_name = f"shm_{cam_id}"
        self.frame_buffer = FrameBuffer(self.shm_name, shape=shape, create=True)

        self.static_zones = []

    def draw_person_roi(self, frame):
        if not self.person_roi:
            return

        x1, y1, x2, y2 = self.person_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def crop_frame(self, frame):
        if not self.crop:
            return frame

        h, w = frame.shape[:2]
        left = self.crop.get("left", 0)
        right = self.crop.get("right", 0)
        top = self.crop.get("top", 0)
        bottom = self.crop.get("bottom", 0)

        return frame[top : h - bottom, left : w - right]

    def expand_bbox(self, bbox, pad_ratio, w, h):
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)

        return [
            max(0, x1 - px),
            max(0, y1 - py),
            min(w, x2 + px),
            min(h, y2 + py),
        ]

    def apply_static_zones(self, frame):
        for z in self.static_zones:
            x1, y1, x2, y2 = z["bbox"]
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            mean = roi.mean(axis=(0, 1))
            frame[y1:y2, x1:x2] = mean

    def start(self):
        cap = None
        zone_sub = None
        if self.zone_port:
            zone_sub = get_sub_socket(self.zone_port)

        while self.running:
            try:
                if cap is not None:
                    cap.release()
                    cap = None
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    print(
                        f"[{self.cam_id}] 카메라 연결 실패 → {self.RETRY_OFFLINE_SEC}초 후 재시도"
                    )
                    time.sleep(self.RETRY_OFFLINE_SEC)
                    continue
                print(f"[{self.cam_id}] Camera Worker 시작…")

                while self.running:
                    if zone_sub:
                        try:
                            msg = zone_sub.recv_json(flags=zmq.NOBLOCK)
                            if msg.get("type") == "static_zone":
                                h, w = self.frame_buffer.shape[:2]
                                bbox = self.expand_bbox(
                                    msg["bbox"], msg["pad_ratio"], w, h
                                )
                                print("오탐영역발견")
                                self.static_zones.append({"bbox": bbox})
                            elif msg.get("type") == "reset_zone":
                                self.static_zones.clear()

                        except zmq.Again:
                            pass

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print(f"[{self.cam_id}] 프레임 읽기 실패 → 카메라 재연결 시도")
                        time.sleep(self.RETRY_DISCONNECT_SEC)
                        break

                    try:
                        frame = self.crop_frame(frame)
                        # if self.cam_id == "shooter1":
                        #    self.draw_person_roi(frame)
                        self.apply_static_zones(frame)
                        self.frame_buffer.write(frame)
                    except Exception as e:
                        print(f"[{self.cam_id}] 프레임 처리 실패: {e}")

            except KeyboardInterrupt:
                print(f"[{self.cam_id}] 사용자 중단")
                break
            except Exception as e:
                print(f"[{self.cam_id}] 오류 발생: {e}")
                print(traceback.format_exc())

                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                time.sleep(self.RETRY_DISCONNECT_SEC)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        print(f"[{self.cam_id}] Camera Worker 종료")
