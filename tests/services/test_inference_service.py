from src.events.events import make_event
from src.events.topics import IMAGE_SUBMITTED, INFERENCE_COMPLETED, PROCESSING_FAILED
from src.services.inference_service import InferenceService


class FakeBroker:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


class FakeBackend:
    def run(self, image_path):
        return {
            "tags": [{"label": "dog", "score": 0.9}],
            "embedding": [0.1, 0.2, 0.3],
            "model_name": "fake-backend",
        }


def test_inference_service_publishes_completed_event():
    broker = FakeBroker()
    backend = FakeBackend()
    service = InferenceService(broker, backend)

    event = make_event(
        IMAGE_SUBMITTED,
        {
            "image_id": "img_001",
            "image_path": "images/dog.jpg",
            "source": "cli",
        },
    )

    service.handle_image_submitted(event)

    assert len(broker.published) == 1

    topic, out_event = broker.published[0]
    assert topic == INFERENCE_COMPLETED
    assert out_event["topic"] == INFERENCE_COMPLETED
    assert out_event["payload"]["image_id"] == "img_001"
    assert out_event["payload"]["image_path"] == "images/dog.jpg"
    assert out_event["payload"]["model_name"] == "fake-backend"


def test_inference_service_rejects_missing_image_path():
    broker = FakeBroker()
    backend = FakeBackend()
    service = InferenceService(broker, backend)

    bad_event = make_event(
        IMAGE_SUBMITTED,
        {
            "image_id": "img_001",
            "source": "cli",
        },
    )

    service.handle_image_submitted(bad_event)
    assert len(broker.published) == 1
    assert broker.published[0][0] == PROCESSING_FAILED
