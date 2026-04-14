from src.events.events import make_event
from src.events.topics import (
    INFERENCE_COMPLETED,
    ANNOTATION_STORED,
    EMBEDDING_STORED,
    PROCESSING_FAILED
)
from src.broker.redis_broker import RedisBroker
from src.stores.document_store import DocumentStore
from src.stores.vector_store import VectorStore


class StorageService:
    def __init__(self, broker, document_store=None, vector_store=None):
        self.broker = broker
        self.document_store = document_store or DocumentStore()
        self.vector_store = vector_store or VectorStore()

    def handle_inference_completed(self, event):
        try:
            payload = event["payload"]

            required_keys = ["image_id", "image_path", "tags", "embedding", "model_name"]
            for key in required_keys:
                if key not in payload:
                    raise ValueError(f"inference.completed missing payload key: {key}")

            image_id = payload["image_id"]

            print(f"[StorageService] Received inference.completed for {image_id}")

            document = {
                "image_id": image_id,
                "image_path": payload["image_path"],
                "tags": payload["tags"],
                "model_name": payload["model_name"],
                "status": "stored",
            }

            self.document_store.save_annotation(image_id, document)
            print(f"[StorageService] Stored document for {image_id}")

            annotation_event = make_event(
                ANNOTATION_STORED,
                {
                    "image_id": image_id,
                    "image_path": payload["image_path"],
                    "tags": payload["tags"],
                },
            )
            self.broker.publish(ANNOTATION_STORED, annotation_event)
            print(f"[StorageService] Published {ANNOTATION_STORED} for {image_id}")

            self.vector_store.save_embedding(image_id, payload["embedding"])
            print(f"[StorageService] Stored embedding for {image_id}")

            embedding_event = make_event(
                EMBEDDING_STORED,
                {
                    "image_id": image_id,
                    "model_name": payload["model_name"],
                    "embedding_dim": len(payload["embedding"]),
                },
            )
            self.broker.publish(EMBEDDING_STORED, embedding_event)
            print(f"[StorageService] Published {EMBEDDING_STORED} for {image_id}")

        except Exception as e:
            print(f"[StorageService] Error processing event: {e}")

            error_event = make_event(
                PROCESSING_FAILED,
                {
                    "service": "StorageService",
                    "operation": "store_annotation_and_embedding",
                    "image_id": payload.get("image_id"),
                    "error": str(e),
                },
            )
            self.broker.publish(PROCESSING_FAILED, error_event)

    def start(self):
        print(f"[StorageService] Listening on topic: {INFERENCE_COMPLETED}")
        self.broker.subscribe(INFERENCE_COMPLETED, self.handle_inference_completed)


if __name__ == "__main__":
    broker = RedisBroker()
    service = StorageService(broker)
    service.start()
