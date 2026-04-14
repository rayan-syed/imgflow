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

        print(f"[InferenceBackend] Loading model: {model_name}")
        print(f"[InferenceBackend] Using device: {self.device}")

        self.labels = LABELS
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"[InferenceBackend] Model loaded successfully")

    def run(self, image_path):
        print(f"[InferenceBackend] Opening image: {image_path}")
        image = Image.open(image_path).convert("RGB")

        print(f"[InferenceBackend] Preparing CLIP inputs")
        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print(f"[InferenceBackend] Running model inference")
        with torch.no_grad():
            outputs = self.model(**inputs)

            image_embeds = outputs.image_embeds[0]
            text_embeds = outputs.text_embeds

            image_embedding = image_embeds.cpu().tolist()

            logits = self.model.logit_scale.exp() * (image_embeds @ text_embeds.T)
            probs = logits.softmax(dim=-1).cpu().tolist()

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
