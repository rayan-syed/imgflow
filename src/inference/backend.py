import warnings
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers.utils import logging as hf_logging

from src.inference.labels import LABELS

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


class InferenceBackend:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, top_k=3):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.labels = LABELS

        print(f"[InferenceBackend] Loading model: {model_name}")
        print(f"[InferenceBackend] Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print("[InferenceBackend] Model loaded successfully")

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _normalize_embedding(self, embedding_tensor):
        embedding_tensor = embedding_tensor / embedding_tensor.norm(dim=-1, keepdim=True)
        return embedding_tensor[0].cpu().tolist()

    def encode_image(self, image_path):
        print(f"[InferenceBackend] Opening image: {image_path}")
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        inputs = self._to_device(inputs)

        print("[InferenceBackend] Encoding image")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return self._normalize_embedding(image_features.pooler_output)

    def encode_text(self, text):
        print(f"[InferenceBackend] Encoding query text: {text}")

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
        inputs = self._to_device(inputs)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return self._normalize_embedding(text_features.pooler_output)

    def run(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = self._to_device(inputs)

        print("[InferenceBackend] Running image-tag inference")
        with torch.no_grad():
            outputs = self.model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embedding = self._normalize_embedding(image_embeds)

            logits = self.model.logit_scale.exp() * (image_embeds @ text_embeds.T)
            probs = logits.softmax(dim=-1)[0].cpu().tolist()

        scored = list(zip(self.labels, probs))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_tags = [
            {"label": label, "score": float(score)}
            for label, score in scored[: self.top_k]
        ]

        print(f"[InferenceBackend] Top tags: {top_tags}")

        return {
            "tags": top_tags,
            "embedding": image_embedding,
            "model_name": self.model_name,
        }
