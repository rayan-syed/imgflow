from src.stores.vector_store import VectorStore


def test_save_and_get_embedding():
    store = VectorStore()

    embedding = [0.1, 0.2, 0.3]
    store.save_embedding("img_001", embedding)

    assert store.has_image("img_001") is True
    assert store.get_embedding("img_001") == embedding
