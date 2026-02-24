import numpy as np
from insightface.app import FaceAnalysis


class FaceEncoder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def encode(self, img):
        faces = self.app.get(img)

        if not faces:
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        emb = face.embedding.astype(np.float32)
        return emb
