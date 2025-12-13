from models.person_model import PersonModel
from utils.zmq_utils import get_sub_socket, get_pub_socket

import msgpack, cv2, numpy as np, time


class InferencePerson:

    def __init__(self, cam_id: str, sub_port: int, pub_port: int):

        self.person_model = PersonModel()
        self.sub_port = sub_port
        self.pub_port = pub_port
        self.cam_id = cam_id

        self.fps_count = 0
        self.last_log = time.time()

    def start(self):

        retry_count = 0
        MAX_RETRIES = 3

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

                        if state in ("drawing", "idle") and conf >= 0.6:
                            bbox = box.xyxy.cpu().numpy()[0].tolist()
                            persons.append({"state": state, "bbox": bbox, "conf": conf})

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
