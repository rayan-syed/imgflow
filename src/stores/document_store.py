import json
import os


class DocumentStore:
    def __init__(self, filepath="data/documents.json"):
        self.filepath = filepath
        self.documents = self._load()

    def _ensure_parent_dir(self):
        parent = os.path.dirname(self.filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _load(self):
        if not os.path.exists(self.filepath):
            return {}

        with open(self.filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        self._ensure_parent_dir()
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

    def save_annotation(self, image_id, document):
        self.documents[image_id] = document
        self._save()

    def get_annotation(self, image_id):
        return self.documents.get(image_id)

    def has_image(self, image_id):
        return image_id in self.documents
