from brain import get_embedding
from vectorize import search_vector
from configs import DEFAULT_COLLECTION_NAME

def search_documents(query, limit=5) -> list[dict]:
    """
    return [{"title": str, "content": str}, ...]
    """
    vector = get_embedding(query)
    return search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
