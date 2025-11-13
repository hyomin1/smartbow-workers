import cv2, time, os

from utils.zmq_utils import get_pub_socket
from dotenv import load_dotenv


load_dotenv()


PORT = int(os.getenv("CAM3_PORT"))
CAM_ID = os.getenv("CAM3_ID")
SOURCE = os.getenv("CAM3_SOURCE")


def crop_frame(frame):
    h, w = frame.shape[:2]

    x_cut_left = 500
    x_cut_right = 250
    y_cut_bottom = 150

    return frame[: h - y_cut_bottom, x_cut_left : w - x_cut_right]


def main():
    pub = get_pub_socket(PORT)

    cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"[{CAM_ID}] 카메라 연결 실패")
        return

    print(f"[{CAM_ID}] Camera Worker 시작…")

    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"[{CAM_ID}] 프레임 읽기 실패… 재시도")
            time.sleep(0.01)
            continue

        frame = crop_frame(frame)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        jpeg_bytes = buffer.tobytes()

        pub.send_multipart(
            [
                CAM_ID.encode(),
                jpeg_bytes,
            ]
        )


if __name__ == "__main__":
    main()
