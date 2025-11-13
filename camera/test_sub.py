from utils.zmq_utils import get_sub_socket
import cv2
import numpy as np

PORT = 5551


def main():
    socket = get_sub_socket([PORT])
    print("Inference SUB 시작…")

    while True:
        cam_id, jpeg = socket.recv_multipart()
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            print("디코드 실패")
            continue

        print(f"[{cam_id.decode()}] frame received: {frame.shape}")


if __name__ == "__main__":
    main()
