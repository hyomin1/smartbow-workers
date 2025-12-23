import numpy as np
from utils.db import SessionLocal
from models.face_embedding import FaceEmbedding


class FaceEmbeddingCache:
    def __init__(self):
        self.cache = {}

    def load(self):
        db = SessionLocal()
        try:
            rows = db.query(FaceEmbedding.user_id, FaceEmbedding.embedding).all()

            for user_id, emb in rows:
                self.cache.setdefault(user_id,[]).append(
                    np.asarray(emb, dtype=np.float32)
                )
            print(f"[FaceCache] loaded users={len(self.cache)}")
        finally:
            db.close()