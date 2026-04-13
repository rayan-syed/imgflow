import uuid
from datetime import datetime, timezone


def make_event(topic, payload):
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
        "topic": topic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


def validate_event(event):
    required_keys = ["event_id", "topic", "timestamp", "payload"]

    for key in required_keys:
        if key not in event:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(event["payload"], dict):
        raise ValueError("payload must be a dictionary")

    if not event["event_id"]:
        raise ValueError("event_id cannot be empty")

    if not event["topic"]:
        raise ValueError("topic cannot be empty")

    return True
