import cv2, time, msgpack, numpy as np

from utils.zmq_utils import get_pub_socket


class CameraWorker:
    def __init__(self, cam_id: str, source: str, pub_port: int, crop=None):
        self.cam_id = cam_id
        self.source = source
        self.pub_port = pub_port
        self.crop = crop

        self.prev_frame = None
    
    def draw_motion_line(self,frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return frame
        
        diff = cv2.absdiff(self.prev_frame, frame)
        gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=10
        )
        vis = frame.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1,y1), (x2,y2), (0,0,255), 2)
        self.prev_frame = frame
        return vis



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
                if self.cam_id == 'target3':
                    frame = self.draw_motion_line(frame)

                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if not ok:
                    continue

                data = msgpack.packb({"type":"frame","cam_id": self.cam_id, "jpeg": buffer.tobytes()})

                pub.send(data)
