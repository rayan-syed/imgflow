from src.events.events import make_event, validate_event
from src.events.topics import IMAGE_SUBMITTED


def test_make_event():
    event = make_event(IMAGE_SUBMITTED, {"x": 1})

    assert "event_id" in event
    assert event["topic"] == IMAGE_SUBMITTED
    assert validate_event(event) is True


def test_validate_event_missing_payload():
    bad_event = {
        "event_id": "evt_123",
        "topic": IMAGE_SUBMITTED,
        "timestamp": "2026-04-13T12:00:00Z",
    }

    try:
        validate_event(bad_event)
        assert False
    except ValueError as e:
        assert "payload" in str(e)


def test_validate_event_payload_dict():
    bad_event = {
        "event_id": "evt_123",
        "topic": IMAGE_SUBMITTED,
        "timestamp": "2026-04-13T12:00:00Z",
        "payload": "not_a_dict",
    }

    try:
        validate_event(bad_event)
        assert False
    except ValueError as e:
        assert "payload" in str(e)


def test_validate_event_invalid_topic():
    bad_event = {
        "event_id": "evt_123",
        "topic": "invalid.topic",
        "timestamp": "2026-04-13T12:00:00Z",
        "payload": {},
    }

    try:
        validate_event(bad_event)
        assert False
    except ValueError as e:
        assert "Invalid topic" in str(e)
