import numpy as np


class FaceRecognizer:
    def __init__(self, face_cache):
        self.face_cache = face_cache
        self.threshold = 0.35

        self.all_embs = np.array([])
        self.user_ids = []
        self._prepare_matrix()

    def _prepare_matrix(self):
        embs_list = []
        for user_id, embs in self.face_cache.cache.items():
            for emb in embs:
                norm_emb = emb / np.linalg.norm(emb)
                embs_list.append(norm_emb)
                self.user_ids.append(user_id)

        if embs_list:
            self.all_embs = np.array(embs_list)

    def recognize(self, query_emb):
        if len(self.all_embs) == 0:
            return None, 0.0

        query_norm = query_emb / np.linalg.norm(query_emb)
        scores = np.dot(self.all_embs, query_norm)

        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        best_user = self.user_ids[best_idx]

        if best_score < self.threshold:
            return None, best_score

        return best_user, best_score
