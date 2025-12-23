from weights.person_model import PersonModel
from utils.zmq_utils import get_sub_socket, get_pub_socket
from inference.face_cache import FaceEmbeddingCache
from inference.face_recognizer import FaceRecognizer
from inference.face_encoder import FaceEncoder

import msgpack, cv2, numpy as np, time


class InferencePerson:

    def __init__(self, cam_id: str, sub_port: int, pub_port: int, gate_port: int):

        self.person_model = PersonModel()
        self.sub_port = sub_port
        self.pub_port = pub_port
        self.gate_port = gate_port
        self.cam_id = cam_id

        self.fps_count = 0
        self.last_log = time.time()

        self.face_cache = FaceEmbeddingCache()
        self.recognizer = None
        self.face_encoder = FaceEncoder()

        self.last_gate_send_ts = 0.0
        self.GATE_INTERVAL = 0.5

    def start(self):
        self.face_cache.load()
        self.recognizer = FaceRecognizer(self.face_cache)


        retry_count = 0
        MAX_RETRIES = 3

        

        print(
            f"[InferencePerson] cam_id={self.cam_id} "
            f"face_users={len(self.face_cache.cache)}"
        )

        while retry_count < MAX_RETRIES:
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
            sub = get_sub_socket(self.sub_port)
            pub = get_pub_socket(self.pub_port)
            gate_pub = get_pub_socket(self.gate_port)
        except Exception as e:
            print(f"[{self.cam_id}] ZMQ 연결 실패: {e}")
            raise
        print(
            f"[{self.cam_id}] InferencePerson 시작 (SUB:{self.sub_port} → PUB:{self.pub_port})"
        )

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10

        while True:
            try:
                data = sub.recv()
                msg = msgpack.unpackb(data, raw=False)

                cam_id = msg["cam_id"]
                jpeg = msg["jpeg"]

                if not cam_id or not jpeg:
                    print("[WARN] Empty cam_id or jpeg")
                    consecutive_errors += 1
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )

                if frame is None:
                    print("[WARN] JPEG decode 실패")
                    consecutive_errors += 1
                    continue
                result = self.person_model.predict(frame)[0]

                persons = []
                for box in result.boxes:
                    try:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        state = result.names[cls]

                        if state in ("drawing") and conf >= 0.7:
                            bbox = box.xyxy.cpu().numpy()[0].tolist()
                            persons.append({"state": state, "bbox": bbox, "conf": conf})

                            now = time.time()

                            if now - self.last_gate_send_ts > self.GATE_INTERVAL:
                                gate_pub.send_json({
                                    "type": "person_gate",
                                    "cam_id": self.cam_id,
                                    "active": True,
                                    "timestamp":now,
                                })
                                self.last_gate_send_ts = now
                            #persons.append({"state": state, "bbox": bbox, "conf": conf, "user_id":user_id, "face_score":face_score})
                            # x1, y1, x2, y2 = map(int, bbox)
                            
                            # person_crop = frame[y1:y2, x1:x2]
                            # user_id = 'unknown',
                            # face_score = 0.0

                            # try:
                            #     face_emb = self.face_encoder.encode(person_crop)
                            #     user_id, face_score = self.recognizer.recognize(face_emb)
                                
                            #     print(f"[FaceTest] cam={cam_id} user={user_id} score={face_score:.3f}")
                            # except Exception as e:
                            #     pass


                            # persons.append({"state": state, "bbox": bbox, "conf": conf, "user_id":user_id, "face_score":face_score})

                    except Exception as e:
                        print(f"[{self.cam_id}] Box 파싱 실패: {e}")
                        continue
                if persons:
                    pub.send_json(
                        {
                            "type": "persons",
                            "cam_id": cam_id,
                            "persons": persons,
                            "timestamp": time.time(),
                        }
                    )

            except msgpack.exceptions.UnpackException as e:
                consecutive_errors += 1
                print(
                    f"[{self.cam_id}] msgpack unpack 실패 ({consecutive_errors}회): {e}"
                )

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"[{self.cam_id}] 연속 에러 {MAX_CONSECUTIVE_ERRORS}회 → 워커 재시작"
                    )
                    raise RuntimeError("연속 에러 임계값 초과")

            except cv2.error as e:
                consecutive_errors += 1
                print(f"[{self.cam_id}] OpenCV 에러 ({consecutive_errors}회): {e}")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"[{self.cam_id}] 연속 에러 {MAX_CONSECUTIVE_ERRORS}회 → 워커 재시작"
                    )
                    raise RuntimeError("연속 에러 임계값 초과")

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
