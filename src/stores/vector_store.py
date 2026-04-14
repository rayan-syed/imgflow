import json
import os

import faiss
import numpy as np


class VectorStore:
    def __init__(
        self,
        dim=512,
        index_path="data/faiss.index",
        ids_path="data/vector_ids.json",
    ):
        self.dim = dim
        self.index_path = index_path
        self.ids_path = ids_path

        self.index = self._load_index()
        self.image_ids = self._load_ids()

    def _ensure_parent_dirs(self):
        for path in [self.index_path, self.ids_path]:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def _load_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)

        return faiss.IndexFlatIP(self.dim)

    def _load_ids(self):
        if os.path.exists(self.ids_path):
            with open(self.ids_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return []

    def _save_index(self):
        self._ensure_parent_dirs()
        faiss.write_index(self.index, self.index_path)

    def _save_ids(self):
        self._ensure_parent_dirs()
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.image_ids, f, indent=2)

    def _normalize(self, embedding):
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)

        if norm == 0:
            return vec

        return vec / norm

    def save_embedding(self, image_id, embedding):
        if self.has_image(image_id):
            return

        vec = self._normalize(embedding).reshape(1, -1)

        if vec.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {vec.shape[1]}"
            )

        self.index.add(vec)
        self.image_ids.append(image_id)

        self._save_index()
        self._save_ids()

    def has_image(self, image_id):
        return image_id in self.image_ids

    def get_count(self):
        return len(self.image_ids)

    def search(self, query_embedding, top_k):
        if self.index.ntotal == 0:
            return []

        query_vec = self._normalize(query_embedding).reshape(1, -1)

        if query_vec.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dim}, got {query_vec.shape[1]}"
            )

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append((self.image_ids[idx], float(score)))

        return results
