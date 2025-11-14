import cv2, time
from utils.zmq_utils import get_pub_socket


class CameraWorker:
    def __init__(self, cam_id: str, source: str, pub_port: int, crop=None):
        self.cam_id = cam_id
        self.source = source
        self.pub_port = pub_port
        self.crop = crop

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
        pub = get_pub_socket(self.pub_port)

        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"[{self.cam_id}] 카메라 연결 실패")
            return
        print(f"[{self.cam_id}] Camera Worker 시작…")

        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[{self.cam_id}] 프레임 읽기 실패… 재시도")
                time.sleep(0.01)
                continue

            frame = self.crop_frame(frame)

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ok:
                continue

            pub.send_multipart(
                [
                    self.cam_id.encode(),
                    buffer.tobytes(),
                ]
            )
