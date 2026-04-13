from src.events.events import make_event
from src.events.topics import INFERENCE_COMPLETED, ANNOTATION_STORED
from src.broker.redis_broker import RedisBroker
from src.stores.document_store import DocumentStore
from src.stores.vector_store import VectorStore


class StorageService:
    def __init__(self, broker, document_store=None, vector_store=None):
        self.broker = broker
        self.document_store = document_store or DocumentStore()
        self.vector_store = vector_store or VectorStore()

    def handle_inference_completed(self, event):
        payload = event["payload"]

        required_keys = ["image_id", "image_path",
                         "tags", "embedding", "model_name"]
        for key in required_keys:
            if key not in payload:
                raise ValueError(
                    f"inference.completed missing payload key: {key}")

        image_id = payload["image_id"]

        document = {
            "image_id": image_id,
            "image_path": payload["image_path"],
            "tags": payload["tags"],
            "model_name": payload["model_name"],
            "status": "stored",
        }

        self.document_store.save_annotation(image_id, document)
        self.vector_store.save_embedding(image_id, payload["embedding"])

        out_event = make_event(
            ANNOTATION_STORED,
            {
                "image_id": image_id,
                "status": "stored",
            },
        )
        self.broker.publish(ANNOTATION_STORED, out_event)

    def start(self):
        self.broker.subscribe(INFERENCE_COMPLETED,
                              self.handle_inference_completed)


if __name__ == "__main__":
    broker = RedisBroker()
    service = StorageService(broker)
    service.start()
