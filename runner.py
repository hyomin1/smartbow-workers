import glob
import os
import signal
import sys
from multiprocessing import Process, shared_memory

from camera.camera_worker import CameraWorker
from config import CAMERA_CONFIG, INFER_ARROW_CONFIG, INFER_PERSON_CONFIG
from inference.inference_arrow import InferenceArrow
from inference.inference_person import InferencePerson


def cleanup_ipc_files():
    ipc_files = glob.glob("/tmp/*.ipc")
    for file in ipc_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


def cleanup_shm():
    for cam_key in CAMERA_CONFIG.keys():
        shm_name = f"shm_{CAMERA_CONFIG[cam_key]['id']}"
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def start_camera_worker(cam_key):
    camera = CAMERA_CONFIG[cam_key]

    kwargs = dict(
        cam_id=camera["id"],
        source=camera["source"],
        shape=camera["shape"],
        crop=camera.get("crop"),
    )
    if camera.get("zone_port"):
        kwargs["zone_port"] = camera["zone_port"]

    worker = CameraWorker(**kwargs)
    worker.start()


def start_inference_worker(cam_key):
    camera = INFER_ARROW_CONFIG[cam_key]

    worker = InferenceArrow(
        cam_id=camera["id"],
        pub_port=camera["infer_port"],
        gate_port=camera["gate_port"],
        target_port=camera["target_port"],
        zone_port=camera["zone_port"],
        shape=camera["shape"],
    )
    worker.start()


def start_inference_person_worker(cam_key):
    camera = INFER_PERSON_CONFIG[cam_key]

    worker = InferencePerson(
        cam_id=camera["id"],
        pub_port=camera["infer_port"],
        gate_port=camera["gate_port"],
        shape=camera["shape"],
    )
    worker.start()


processes = []


def handle_exit(signum, frame):
    print("Received exit signal. Terminating child processes...")
    for p in processes:
        if p.is_alive():
            print(f"Terminating process {p.pid}...")
            p.terminate()

    # Wait for all processes to terminate
    for p in processes:
        print(f"Waiting for process {p.pid} to join...")
        p.join()  # Remove the short timeout

    print("All processes terminated. Cleaning up shared memory...")
    cleanup_shm()
    print("Cleanup complete. Exiting.")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    cleanup_ipc_files()
    cleanup_shm()

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
