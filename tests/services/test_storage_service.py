from src.events.events import make_event
from src.events.topics import INFERENCE_COMPLETED, ANNOTATION_STORED
from src.services.storage_service import StorageService
from src.stores.document_store import DocumentStore
from src.stores.vector_store import VectorStore


class FakeBroker:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


def test_storage_service_stores_outputs_and_publishes_annotation_stored():
    broker = FakeBroker()
    document_store = DocumentStore()
    vector_store = VectorStore()

    service = StorageService(broker, document_store, vector_store)

    event = make_event(
        INFERENCE_COMPLETED,
        {
            "image_id": "img_001",
            "image_path": "images/dog.jpg",
            "tags": [{"label": "dog", "score": 0.9}],
            "embedding": [0.1, 0.2, 0.3],
            "model_name": "dummy-backend",
        },
    )

    service.handle_inference_completed(event)

    assert document_store.has_image("img_001") is True
    assert vector_store.has_image("img_001") is True

    stored_doc = document_store.get_annotation("img_001")
    assert stored_doc["image_id"] == "img_001"
    assert stored_doc["image_path"] == "images/dog.jpg"

    stored_embedding = vector_store.get_embedding("img_001")
    assert stored_embedding == [0.1, 0.2, 0.3]

    assert len(broker.published) == 1
    topic, out_event = broker.published[0]
    assert topic == ANNOTATION_STORED
    assert out_event["topic"] == ANNOTATION_STORED
    assert out_event["payload"]["image_id"] == "img_001"