import json
import uuid

from src.broker.redis_broker import RedisBroker
from src.events.events import make_event
from src.events.topics import (
    IMAGE_SUBMITTED,
    QUERY_SUBMITTED,
    ANNOTATION_STORED,
    EMBEDDING_STORED,
    QUERY_COMPLETED,
    PROCESSING_FAILED,
)


class CLI:
    def __init__(self):
        self.broker = RedisBroker()

    def print_help(self):
        print("\nAvailable Commands:")
        print("  help                 Show available commands")
        print("  upload <image_path>  Submit an image for inference")
        print("  query <text>         Search for similar stored images")
        print("  exit                 Exit the CLI\n")

    def wait_for_upload_events(self, image_id):
        pubsub = self.broker.client.pubsub()
        pubsub.subscribe(ANNOTATION_STORED, EMBEDDING_STORED, PROCESSING_FAILED)

        annotation_seen = False
        embedding_seen = False

        try:
            for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                event = json.loads(message["data"])
                topic = event["topic"]
                payload = event["payload"]

                if topic == PROCESSING_FAILED and payload.get("image_id") == image_id:
                    print(f"\nUpload failed for image: {image_id}")
                    print(f"Service: {payload.get('service')}")
                    print(f"Operation: {payload.get('operation')}")
                    print(f"Error: {payload.get('error')}")
                    break

                if payload.get("image_id") != image_id:
                    continue

                if topic == ANNOTATION_STORED:
                    print(f"\nAnnotation stored for image: {image_id}")
                    print(f"Tags: {payload.get('tags', [])}")
                    annotation_seen = True

                elif topic == EMBEDDING_STORED:
                    print(f"\nEmbedding stored for image: {image_id}")
                    print(f"Embedding Dimension: {payload.get('embedding_dim')}")
                    embedding_seen = True

                if annotation_seen and embedding_seen:
                    break
        finally:
            pubsub.close()

    def wait_for_query_completed(self, query_id):
        pubsub = self.broker.client.pubsub()
        pubsub.subscribe(QUERY_COMPLETED, PROCESSING_FAILED)

        try:
            for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                event = json.loads(message["data"])
                topic = event["topic"]
                payload = event["payload"]

                if topic == PROCESSING_FAILED and payload.get("query_id") == query_id:
                    print(f"\nQuery failed: {query_id}")
                    print(f"Service: {payload.get('service')}")
                    print(f"Operation: {payload.get('operation')}")
                    print(f"Error: {payload.get('error')}")
                    break

                if topic != QUERY_COMPLETED:
                    continue

                if payload.get("query_id") != query_id:
                    continue

                results = payload.get("results", [])

                print(f"\nQuery completed: {query_id}")

                if not results:
                    print("No results found.")
                    break

                print("Results:")
                for i, result in enumerate(results, 1):
                    image_id = result.get("image_id")
                    score = result.get("score", 0.0)
                    document = result.get("document", {})
                    tags = document.get("tags", [])

                    print(f"{i}. Image ID: {image_id}, Score: {score:.4f}")
                    print(f"   Tags: {tags}")

                break
        finally:
            pubsub.close()

    def submit_image(self, image_path):
        image_id = f"img_{uuid.uuid4().hex[:8]}"

        event = make_event(
            IMAGE_SUBMITTED,
            {
                "image_id": image_id,
                "image_path": image_path,
                "source": "cli",
            },
        )

        self.broker.publish(IMAGE_SUBMITTED, event)

        print(f"Submitted image: {image_path}")
        print(f"Image ID: {image_id}, Image Path: {image_path}")

        self.wait_for_upload_events(image_id)

    def submit_query(self, query_text, top_k=3):
        query_id = f"qry_{uuid.uuid4().hex[:8]}"

        event = make_event(
            QUERY_SUBMITTED,
            {
                "query_id": query_id,
                "query_text": query_text,
                "top_k": top_k,
            },
        )

        self.broker.publish(QUERY_SUBMITTED, event)

        print(f"Submitted query: {query_text}")
        print(f"Query ID: {query_id}, Query Text: {query_text}, Top K: {top_k}")

        self.wait_for_query_completed(query_id)

    def run(self):
        print("=== ImgFlow CLI ===")
        print("type 'help' to see available commands\n")

        while True:
            try:
                raw = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting ImgFlow CLI.")
                break

            if not raw:
                continue

            if raw == "exit":
                print("Exiting ImgFlow CLI.")
                break

            if raw == "help":
                self.print_help()
                continue

            parts = raw.split(maxsplit=1)
            command = parts[0]

            if command == "upload":
                if len(parts) < 2:
                    print("Usage: upload <image_path>")
                    continue

                self.submit_image(parts[1])

            elif command == "query":
                if len(parts) < 2:
                    print("Usage: query <query_text>")
                    continue

                self.submit_query(parts[1])

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' to see available commands.")


if __name__ == "__main__":
    CLI().run()
