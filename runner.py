from config import CAMERA_CONFIG, INFER_ARROW_CONFIG, INFER_PERSON_CONFIG
from camera.camera_worker import CameraWorker
from multiprocessing import Process
from inference.inference_arrow import InferenceArrow
from inference.inference_person import InferencePerson




def start_camera_worker(cam_key):
    camera = CAMERA_CONFIG[cam_key]

    worker = CameraWorker(
        cam_id=camera["id"],
        source=camera["source"],
        pub_port=camera["raw_port"],
        crop=camera["crop"],
    )
    worker.start()


def start_inference_worker(cam_key):
    camera = INFER_ARROW_CONFIG[cam_key]

    worker = InferenceArrow(
        cam_id=camera["id"],
        sub_port=camera["raw_port"],
        pub_port=camera["infer_port"],
        gate_port=camera['gate_port'],
    )
    worker.start()


def start_inference_person_worker(cam_key):
    camera = INFER_PERSON_CONFIG[cam_key]

    worker = InferencePerson(
        cam_id=camera["id"],
        sub_port=camera["raw_port"],
        pub_port=camera["infer_port"],
        gate_port=camera['gate_port'],
    )
    worker.start()


def main():
    processes = []

    for cam_key in CAMERA_CONFIG.keys():
        p = Process(target=start_camera_worker, args=(cam_key,))
        p.start()
        processes.append(p)

    for cam_key in INFER_ARROW_CONFIG.keys():
        p = Process(target=start_inference_worker, args=(cam_key,))
        p.start()
        processes.append(p)

    for cam_key in INFER_PERSON_CONFIG.keys():
        p = Process(target=start_inference_person_worker, args=(cam_key,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
