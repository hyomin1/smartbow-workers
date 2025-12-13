import os
from dotenv import load_dotenv

load_dotenv()

CAMERA_CONFIG = {
    # "target1": {
    #     "id": os.getenv("TARGET1_ID"),
    #     "source": os.getenv("TARGET1_SOURCE"),
    #     "raw_port": int(os.getenv("TARGET1_PORT")),
    #     "crop": None,
    # },
    # "target2": {
    #     "id": os.getenv("TARGET2_ID"),
    #     "source": os.getenv("TARGET2_SOURCE"),
    #     "raw_port": int(os.getenv("TARGET2_PORT")),
    #     "crop": None,
    # },
    "target3": {
        "id": os.getenv("TARGET3_ID"),
        "source": os.getenv("TARGET3_SOURCE"),
        "raw_port": int(os.getenv("TARGET3_PORT")),
        "crop": {"left": 550, "right": 250, "top": 0, "bottom": 200},
    },
    # "target-test": {
    #     "id": os.getenv("TARGET_ID"),
    #     "source": os.getenv("TARGET_SOURCE"),
    #     "raw_port": int(os.getenv("TARGET_PORT")),
    #     "crop": {"left": 500, "right": 250, "top": 0, "bottom": 150},
    # },
    "shooter-test": {
        "id": os.getenv("SHOOTER_ID"),
        "source": os.getenv("SHOOTER_SOURCE"),
        "raw_port": int(os.getenv("SHOOTER_PORT")),
        "crop": None,
    },
    # "shooter1": {
    #     "id": os.getenv("SHOOTER1_ID"),
    #     "source": os.getenv("SHOOTER1_SOURCE"),
    #     "raw_port": int(os.getenv("SHOOTER1_PORT")),
    #     "crop": None,
    # },
    # "shooter2": {
    #     "id": os.getenv("SHOOTER2_ID"),
    #     "source": os.getenv("SHOOTER2_SOURCE"),
    #     "raw_port": int(os.getenv("SHOOTER2_PORT")),
    #     "crop": None,
    # },
}

INFER_ARROW_CONFIG = {
    # "target1": {
    #     "id": os.getenv("TARGET1_ID"),
    #     "infer_port": int(os.getenv("TARGET1_INFER_PORT")),
    #     "raw_port": int(os.getenv("TARGET1_PORT")),
    # },
    # "target2": {
    #     "id": os.getenv("TARGET2_ID"),
    #     "infer_port": int(os.getenv("TARGET2_INFER_PORT")),
    #     "raw_port": int(os.getenv("TARGET2_PORT")),
    # },
    "target3": {
        "id": os.getenv("TARGET3_ID"),
        "infer_port": int(os.getenv("TARGET3_INFER_PORT")),
        "raw_port": int(os.getenv("TARGET3_PORT")),
    },
    # "target-test": {
    #     "id": os.getenv("TARGET_ID"),
    #     "infer_port": int(os.getenv("TARGET_INFER_PORT")),
    #     "raw_port": int(os.getenv("TARGET_PORT")),
    # },
}
INFER_PERSON_CONFIG = {
    "shooter-test": {
        "id": os.getenv("SHOOTER_ID"),
        "infer_port": int(os.getenv("SHOOTER_INFER_PORT")),
        "raw_port": int(os.getenv("SHOOTER_PORT")),
    },
    # "shooter1": {
    #     "id": os.getenv("SHOOTER1_ID"),
    #     "infer_port": int(os.getenv("SHOOTER1_INFER_PORT")),
    #     "raw_port": int(os.getenv("SHOOTER1_PORT")),
    # },
    # "shooter2": {
    #     "id": os.getenv("SHOOTER2_ID"),
    #     "infer_port": int(os.getenv("SHOOTER2_INFER_PORT")),
    #     "raw_port": int(os.getenv("SHOOTER2_PORT")),
    # },
}
