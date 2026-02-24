import time

from inference.face_cache import FaceEmbeddingCache
from inference.face_encoder import FaceEncoder
from inference.face_recognizer import FaceRecognizer
from utils.frame_shm import FrameBuffer
from utils.zmq_utils import get_pub_socket
from weights.person_model import PersonModel
from workers.base import BaseWorker


class InferencePerson(BaseWorker):
    def __init__(self, cam_id: str, pub_port: str, gate_port: str, shape):
        super().__init__(f"InferencePerson-{cam_id}")
        self.person_model = PersonModel()
        self.pub_port = pub_port
        self.gate_port = gate_port
        self.cam_id = cam_id

        self.shm_name = f"shm_{cam_id}"
        self.frame_buffer = FrameBuffer(name=self.shm_name, shape=shape, create=False)

        self.fps_count = 0
        self.last_log = time.time()

        self.face_cache = FaceEmbeddingCache()
        self.recognizer = None
        self.face_encoder = FaceEncoder()

        self.last_gate_send_ts = 0.0
        self.GATE_INTERVAL = 0.5

        self.person_roi = (
            0,
            320,
            800,
            820,
        )  # 해당 영역만 추론하기 위함 (3관 전용 추후 확장)

        self.last_infer_ts = 0.0
        self.INFER_INTERVAL = 0.2

    def start(self):
        self.face_cache.load()
        self.recognizer = FaceRecognizer(self.face_cache)

        retry_count = 0
        MAX_RETRIES = 3

        print(
            f"[InferencePerson] cam_id={self.cam_id} "
            f"face_users={len(self.face_cache.cache)}"
        )

        while self.running and retry_count < MAX_RETRIES:
            try:
                self.run()
            except Exception as e:
                retry_count += 1
                print(f"[{self.cam_id}]에러 발생 ({retry_count}/{MAX_RETRIES}): {e}")

                if retry_count < MAX_RETRIES:
                    wait_time = retry_count * 5
                    print(f"[{self.cam_id}] {wait_time}초 후 재시도…")
                    time.sleep(wait_time)
                else:
                    print(f"[{self.cam_id}] 재시도 실패")
                    raise

    def run(self):
        try:
            pub = get_pub_socket(self.pub_port)
            gate_pub = get_pub_socket(self.gate_port)
        except Exception as e:
            print(f"[{self.cam_id}] ZMQ 연결 실패: {e}")
            raise
        print(f"[{self.cam_id}] InferencePerson 시작 ")

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10

        while self.running:
            try:
                frame = self.frame_buffer.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                if now - self.last_infer_ts < self.INFER_INTERVAL:
                    time.sleep(0.005)
                    continue
                self.last_infer_ts = time.time()

                x1, y1, x2, y2 = self.person_roi
                roi_frame = frame[y1:y2, x1:x2]

                person = self.person_model.predict(roi_frame)
                if not person:
                    continue

                bx1, by1, bx2, by2 = map(int, person["bbox"])
                bx1 += x1
                bx2 += x1
                by1 += y1
                by2 += y1

                person["bbox"] = [bx1, by1, bx2, by2]
                person_crop = frame[by1:by2, bx1:bx2]
                face_emb = self.face_encoder.encode(person_crop)

                user_id = None
                if face_emb is not None and self.recognizer:
                    user_id, _ = self.recognizer.recognize(face_emb)
                person["user_id"] = user_id

                ts = time.time()
                if ts - self.last_gate_send_ts > self.GATE_INTERVAL:
                    self.last_gate_send_ts = ts
                    gate_pub.send_json(
                        {
                            "type": "person_gate",
                            "cam_id": self.cam_id,
                            "active": True,
                            "timestamp": ts,
                        }
                    )
                pub.send_json(
                    {
                        "type": "person",
                        "cam_id": self.cam_id,
                        "person": person,
                        "timestamp": ts,
                    }
                )

            except KeyboardInterrupt:
                print(f"[{self.cam_id}] 사용자 중단")
                break

            except Exception as e:
                consecutive_errors += 1
                print(
                    f"[{self.cam_id}] 예상치 못한 에러 ({consecutive_errors}회): {type(e).__name__} - {e}"
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"[{self.cam_id}] 연속 에러 {MAX_CONSECUTIVE_ERRORS}회 → 워커 재시작"
                    )
                    raise RuntimeError("연속 에러 임계값 초과")
