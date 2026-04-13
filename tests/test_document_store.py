from src.stores.document_store import DocumentStore


def test_save_and_get_annotation():
    store = DocumentStore()

    doc = {"image_id": "img_001", "tags": [{"label": "dog", "score": 0.9}]}
    store.save_annotation("img_001", doc)

    assert store.has_image("img_001") is True
    assert store.get_annotation("img_001") == doc
