from src.events.events import make_event
from src.events.topics import QUERY_SUBMITTED, QUERY_COMPLETED, PROCESSING_FAILED
from src.services.query_service import QueryService
from src.stores.document_store import DocumentStore
from src.stores.vector_store import VectorStore


class FakeBroker:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


class FakeBackend:
    def encode_text(self, text):
        if text == "dog":
            return [1.0, 0.0, 0.0]
        if text == "cat":
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_query_service_returns_ranked_results(tmp_path):
    broker = FakeBroker()
    document_store = DocumentStore(
        filepath=str(tmp_path / "documents.json")
    )
    vector_store = VectorStore(
        dim=3,
        index_path=str(tmp_path / "faiss.index"),
        ids_path=str(tmp_path / "vector_ids.json"),
    )
    backend = FakeBackend()

    document_store.save_annotation(
        "img_1",
        {
            "image_id": "img_1",
            "image_path": "images/dog1.jpg",
            "tags": [{"label": "dog", "score": 0.95}],
            "model_name": "test-model",
            "status": "stored",
        },
    )
    document_store.save_annotation(
        "img_2",
        {
            "image_id": "img_2",
            "image_path": "images/cat1.jpg",
            "tags": [{"label": "cat", "score": 0.96}],
            "model_name": "test-model",
            "status": "stored",
        },
    )

    vector_store.save_embedding("img_1", [1.0, 0.0, 0.0])
    vector_store.save_embedding("img_2", [0.0, 1.0, 0.0])

    service = QueryService(
        broker=broker,
        document_store=document_store,
        vector_store=vector_store,
        backend=backend,
    )

    event = make_event(
        QUERY_SUBMITTED,
        {
            "query_id": "qry_001",
            "query_text": "dog",
            "top_k": 2,
        },
    )

    service.handle_query_submitted(event)

    assert len(broker.published) == 1

    topic, out_event = broker.published[0]

    assert topic == QUERY_COMPLETED
    assert out_event["topic"] == QUERY_COMPLETED
    assert out_event["payload"]["query_id"] == "qry_001"

    results = out_event["payload"]["results"]
    assert len(results) == 2
    assert results[0]["image_id"] == "img_1"
    assert results[1]["image_id"] == "img_2"

    assert results[0]["document"]["image_id"] == "img_1"
    assert results[0]["document"]["tags"][0]["label"] == "dog"


def test_query_service_rejects_missing_query_text(tmp_path):
    broker = FakeBroker()
    document_store = DocumentStore(
        filepath=str(tmp_path / "documents.json")
    )
    vector_store = VectorStore(
        dim=3,
        index_path=str(tmp_path / "faiss.index"),
        ids_path=str(tmp_path / "vector_ids.json"),
    )
    backend = FakeBackend()

    service = QueryService(
        broker=broker,
        document_store=document_store,
        vector_store=vector_store,
        backend=backend,
    )

    bad_event = make_event(
        QUERY_SUBMITTED,
        {
            "query_id": "qry_002",
            "top_k": 2,
        },
    )

    service.handle_query_submitted(bad_event)

    assert len(broker.published) == 1
    assert broker.published[0][0] == PROCESSING_FAILED
