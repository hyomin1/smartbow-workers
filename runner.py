from config import CAMERA_CONFIG, INFER_CONFIG
from camera.camera_worker import CameraWorker
from multiprocessing import Process
from inference.inference_worker import InferenceWorker


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
    camera = INFER_CONFIG[cam_key]

    worker = InferenceWorker(
        cam_id=camera["id"],
        sub_port=camera["raw_port"],
        pub_port=camera["infer_port"],
    )
    worker.start()


def main():
    processes = []

    for cam_key in CAMERA_CONFIG.keys():
        p = Process(target=start_camera_worker, args=(cam_key,))
        p.start()
        processes.append(p)

    for cam_key in INFER_CONFIG.keys():
        p = Process(target=start_inference_worker, args=(cam_key,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
