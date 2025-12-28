import time

import cv2

from utils.frame_shm import FrameBuffer


class CameraWorker:
    def __init__(self, cam_id: str, source: str, shape=(1080, 1920, 3), crop=None):
        self.cam_id = cam_id
        self.source = source

        self.crop = crop

        self.shm_name = f"shm_{cam_id}"
        self.frame_buffer = FrameBuffer(self.shm_name, shape=shape, create=True)

    def crop_frame(self, frame):
        if not self.crop:
            return frame

        h, w = frame.shape[:2]
        left = self.crop.get("left", 0)
        right = self.crop.get("right", 0)
        top = self.crop.get("top", 0)
        bottom = self.crop.get("bottom", 0)

        return frame[top : h - bottom, left : w - right]

    def start(self):
        while True:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print(f"[{self.cam_id}] 카메라 연결 실패 → 3초 후 재시도")
                time.sleep(3)
                continue

            print(f"[{self.cam_id}] Camera Worker 시작…")

            while True:
                ok, frame = cap.read()

                if not ok or frame is None:
                    print(f"[{self.cam_id}] 프레임 읽기 실패 → 카메라 재연결 시도")
                    cap.release()
                    time.sleep(1)
                    break

                frame = self.crop_frame(frame)

                self.frame_buffer.write(frame)
