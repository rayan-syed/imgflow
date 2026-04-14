from src.events.events import make_event
from src.events.topics import (
    INFERENCE_COMPLETED,
    ANNOTATION_STORED,
    EMBEDDING_STORED,
)
from src.services.storage_service import StorageService
from src.stores.document_store import DocumentStore
from src.stores.vector_store import VectorStore


class FakeBroker:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


def test_storage_service_stores_outputs():
    broker = FakeBroker()
    document_store = DocumentStore()
    vector_store = VectorStore(dim=3)

    service = StorageService(
        broker,
        document_store,
        vector_store,
    )

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

    assert document_store.has_image("img_001")
    assert vector_store.has_image("img_001")

    stored_doc = document_store.get_annotation("img_001")
    assert stored_doc["image_id"] == "img_001"

    results = vector_store.search([0.1, 0.2, 0.3], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "img_001"

    assert len(broker.published) == 2

    topic1, event1 = broker.published[0]
    topic2, event2 = broker.published[1]

    assert topic1 == ANNOTATION_STORED
    assert event1["topic"] == ANNOTATION_STORED
    assert event1["payload"]["image_id"] == "img_001"

    assert topic2 == EMBEDDING_STORED
    assert event2["topic"] == EMBEDDING_STORED
    assert event2["payload"]["image_id"] == "img_001"


def test_storage_service_rejects_missing_embedding():
    broker = FakeBroker()
    document_store = DocumentStore()
    vector_store = VectorStore(dim=3)

    service = StorageService(
        broker,
        document_store,
        vector_store,
    )

    bad_event = make_event(
        INFERENCE_COMPLETED,
        {
            "image_id": "img_001",
            "image_path": "images/test.jpg",
            "tags": [{"label": "dog", "score": 0.9}],
            "model_name": "dummy-backend",
            # missing embedding
        },
    )

    service.handle_inference_completed(bad_event)

    assert document_store.has_image("img_001") is False
    assert vector_store.has_image("img_001") is False
    assert len(broker.published) == 0
