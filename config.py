import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key, default=''):
    return os.getenv(key, default)

CAMERA_CONFIG = {
    # "target1": {
    #     "id": get_env("TARGET1_ID"),
    #     "source": get_env("TARGET1_SOURCE"),
    #     "raw_port": get_env("TARGET1_RAW_IPC"),
    #     "crop": None,
    # },
    # "target2": {
    #     "id": get_env("TARGET2_ID"),
    #     "source": get_env("TARGET2_SOURCE"),
    #     "raw_port": get_env("TARGET2_RAW_IPC"),
    #     "crop": None,
    # },
    "target3": {
        "id": get_env("TARGET3_ID"),
        "source": get_env("TARGET3_SOURCE"),
        "raw_port": get_env("TARGET3_RAW_IPC"),
        "crop": {"left": 450, "right": 350, "top": 0, "bottom": 200},
    },
    # "target-test": {
    #     "id": get_env("TARGET_ID"),
    #     "source": get_env("TARGET_SOURCE"),
    #     "raw_port": get_env("TARGET_INFER_IPC"),
    #     "crop": {"left": 500, "right": 250, "top": 0, "bottom": 150},
    # },
    # "shooter-test": {
    #     "id": get_env("SHOOTER_ID"),
    #     "source": get_env("SHOOTER_SOURCE"),
    #     "raw_port": get_env("SHOOTER1_RAW_IPC"),
    #     "crop": None,
    # },
    "shooter1": {
        "id": get_env("SHOOTER1_ID"),
        "source": get_env("SHOOTER1_SOURCE"),
        "raw_port": get_env("SHOOTER1_RAW_IPC"),
        "crop": {"left": 200, "right": 400, "top": 300, "bottom": 180},
    },
    # "shooter2": {
    #     "id": get_env("SHOOTER2_ID"),
    #     "source": get_env("SHOOTER2_SOURCE"),
    #     "raw_port": get_env("SHOOTER1_RAW_IPC"),
    #     "crop": None,
    # },
}

INFER_ARROW_CONFIG = {
    # "target1": {
    #     "id": get_env("TARGET1_ID"),
    #     "infer_port": get_env("TARGET1_INFER_IPC"),
    #     "raw_port": get_env("TARGET1_RAW_IPC"),
    #     "gate_port": get_env("TARGET1_GATE_IPC"),
    # },
    # "target2": {
    #     "id": get_env("TARGET2_ID"),
    #     "infer_port": (get_env("TARGET2_RAW_IPC")),
    #     "raw_port": (get_env("TARGET2_INFER_IPC")),
    #     "gate_port": (get_env("TARGET2_GATE_IPC")),
    # },
    "target3": {
        "id": get_env("TARGET3_ID"),
        "infer_port": get_env("TARGET3_INFER_IPC"),
        "raw_port": get_env("TARGET3_RAW_IPC"),
        "gate_port": get_env("TARGET3_GATE_IPC")
    },
    # "target-test": {
    #     "id": get_env("TARGET_ID"),
    #     "infer_port": get_env("TARGET_INFER_IPC"),
    #     "raw_port": get_env("TARGET_RAW_IPC"),
    #     "gate_port": get_env("TARGET_GATE_IPC"),
    # },
}
INFER_PERSON_CONFIG = {
    # "shooter-test": {
    #     "id": get_env("SHOOTER_ID"),
    #     "infer_port": (get_env("SHOOTER_INFER_IPC")),
    #     "raw_port": (get_env("SHOOTER_RAW_IPC")),
    #     "gate_port": (get_env("SHOOTER_GATE_IPC")),
    # },
    "shooter1": {
        "id": get_env("SHOOTER1_ID"),
        "infer_port": get_env("SHOOTER1_INFER_IPC"),
        "raw_port": get_env("SHOOTER1_RAW_IPC"),
        "gate_port": get_env("SHOOTER1_GATE_IPC"),
    },
    # "shooter2": {
    #     "id": get_env("SHOOTER2_ID"),
    #     "infer_port": get_env("SHOOTER2_INFER_IPC"),
    #     "raw_port": get_env("SHOOTER2_RAW_IPC"),
    #     "gate_port": get_env("SHOOTER2_GATE_IPC"),
    # },
}

DATABASE_URL = get_env("DATABASE_URL")