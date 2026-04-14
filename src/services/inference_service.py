from src.events.events import make_event
from src.events.topics import IMAGE_SUBMITTED, INFERENCE_COMPLETED
from src.broker.redis_broker import RedisBroker
from src.inference.backend import InferenceBackend


class InferenceService:
    def __init__(self, broker, backend=None):
        self.broker = broker
        self.backend = backend or InferenceBackend()

    def handle_image_submitted(self, event):
        try:
            payload = event["payload"]

            required_keys = ["image_id", "image_path", "source"]
            for key in required_keys:
                if key not in payload:
                    raise ValueError(f"image.submitted missing payload key: {key}")

            print(
                f"[InferenceService] Received image.submitted for {payload['image_id']}")
            print(f"[InferenceService] Running backend on {payload['image_id']}")

            result = self.backend.run(payload["image_path"])

            out_payload = {
                "image_id": payload["image_id"],
                "image_path": payload["image_path"],
                "tags": result["tags"],
                "embedding": result["embedding"],
                "model_name": result["model_name"],
            }

            out_event = make_event(INFERENCE_COMPLETED, out_payload)
            self.broker.publish(INFERENCE_COMPLETED, out_event)

            print(
                f"[InferenceService] Published {INFERENCE_COMPLETED} for {payload['image_id']}")

        except Exception as e:
            print(f"[InferenceService] Error processing event: {e}")

    def start(self):
        print(f"[InferenceService] Listening on topic: {IMAGE_SUBMITTED}")
        self.broker.subscribe(IMAGE_SUBMITTED, self.handle_image_submitted)


if __name__ == "__main__":
    broker = RedisBroker()
    service = InferenceService(broker)
    service.start()
