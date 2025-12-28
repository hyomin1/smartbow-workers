import os

from dotenv import load_dotenv

load_dotenv()


def get_env(key, default=""):
    return os.getenv(key, default)


CAMERA_CONFIG = {
    # "target1": {
    #     "id": get_env("TARGET1_ID"),
    #     "source": get_env("TARGET1_SOURCE"),
    #     "crop": None,
    #     "shape": (1080, 1920, 3),
    # },
    # "target2": {
    #     "id": get_env("TARGET2_ID"),
    #     "source": get_env("TARGET2_SOURCE"),
    #     "crop": None,
    #     "shape": (1080, 1920, 3),
    # },
    # "target3": {
    #     "id": get_env("TARGET3_ID"),
    #     "source": get_env("TARGET3_SOURCE"),
    #     "crop": {"left": 450, "right": 350, "top": 0, "bottom": 200},
    #     "shape": (
    #         1080 - 0 - 200,  # H - top - bottom
    #         1920 - 450 - 350,  # W - left - right
    #         3,
    #     ),
    # },
    # "shooter1": {
    #     "id": get_env("SHOOTER1_ID"),
    #     "source": get_env("SHOOTER1_SOURCE"),
    #     "crop": {"left": 200, "right": 400, "top": 300, "bottom": 180},
    #      "shape": (
    #        1080 - 300 - 180,
    #        1920 - 200 - 400,
    #        3,
    #    ),
    #
    # },
    "target-test": {
        "id": get_env("TARGET_ID"),
        "source": get_env("TARGET_SOURCE"),
        "crop": {"left": 500, "right": 250, "top": 0, "bottom": 150},
        "shape": (
            1080 - 0 - 150,
            1920 - 500 - 250,
            3,
        ),
    },
    "shooter-test": {
        "id": get_env("SHOOTER_ID"),
        "source": get_env("SHOOTER_SOURCE"),
        "crop": None,
        "shape": (1080, 1920, 3),
    },
    # "shooter2": {
    #     "id": get_env("SHOOTER2_ID"),
    #     "source": get_env("SHOOTER2_SOURCE"),
    #     "crop": None,
    #     "shape": (1080, 1920, 3),
    # },
}

INFER_ARROW_CONFIG = {
    # "target1": {
    #     "id": get_env("TARGET1_ID"),
    #     "infer_port": get_env("TARGET1_INFER_IPC"),
    #     "gate_port": get_env("TARGET1_GATE_IPC"),
    #     "shape": (1080, 1920, 3),
    # },
    # "target2": {
    #     "id": get_env("TARGET2_ID"),
    #     "infer_port": (get_env("TARGET2_INFER_IPC")),
    #     "gate_port": (get_env("TARGET2_GATE_IPC")),
    #     "shape": (1080, 1920, 3),
    # },
    # "target3": {
    #     "id": get_env("TARGET3_ID"),
    #     "infer_port": get_env("TARGET3_INFER_IPC"),
    #     "gate_port": get_env("TARGET3_GATE_IPC"),
    #     "shape": (
    #         1080 - 0 - 200,
    #         1920 - 450 - 350,
    #         3,
    #     ),
    # },
    "target-test": {
        "id": get_env("TARGET_ID"),
        "infer_port": get_env("TARGET_INFER_IPC"),
        "gate_port": get_env("TARGET_GATE_IPC"),
        "shape": (
            1080 - 0 - 150,
            1920 - 500 - 250,
            3,
        ),
    },
}
INFER_PERSON_CONFIG = {
    "shooter-test": {
        "id": get_env("SHOOTER_ID"),
        "infer_port": (get_env("SHOOTER_INFER_IPC")),
        "gate_port": (get_env("SHOOTER_GATE_IPC")),
        "shape": (1080, 1920, 3),
    },
    # "shooter1": {
    #     "id": get_env("SHOOTER1_ID"),
    #     "infer_port": get_env("SHOOTER1_INFER_IPC"),
    #     "gate_port": get_env("SHOOTER1_GATE_IPC"),
    #     "shape": (
    #         1080 - 300 - 180,
    #         1920 - 200 - 400,
    #         3,
    #     ),
    # },
    # "shooter2": {
    #     "id": get_env("SHOOTER2_ID"),
    #     "infer_port": get_env("SHOOTER2_INFER_IPC"),
    #     "gate_port": get_env("SHOOTER2_GATE_IPC"),
    #     "shape": (1080, 1920, 3),
    # },
}

DATABASE_URL = get_env("DATABASE_URL")
