import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.image_ids = []

    def normalize(self, embedding):
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def save_embedding(self, image_id, embedding):
        vec = self.normalize(embedding).reshape(1, -1)
        self.index.add(vec)
        self.image_ids.append(image_id)

    def has_image(self, image_id):
        return image_id in self.image_ids

    def get_count(self):
        return len(self.image_ids)

    def search(self, query_embedding, top_k):
        if self.index.ntotal == 0:
            return []

        query_vec = self.normalize(query_embedding).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append((
                self.image_ids[idx],
                float(score),
            ))

        return results
