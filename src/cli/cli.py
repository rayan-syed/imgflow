import uuid

from src.broker.redis_broker import RedisBroker
from src.events.events import make_event
from src.events.topics import IMAGE_SUBMITTED, QUERY_SUBMITTED


class CLI:
    def __init__(self):
        self.broker = RedisBroker()

    def print_help(self):
        print("\nAvailable Commands:")
        print("  help                 Show available commands")
        print("  upload <image_path>  Submit an image for inference")
        print("  query <text>         Search for similar stored images")
        print("  exit                 Exit the CLI\n")

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
