from brain import get_embedding
from vectorize import search_vector_ids
from configs import DEFAULT_COLLECTION_NAME


def search_documents_ids(query, limit=5):
    vector = get_embedding(query)
    return search_vector_ids(DEFAULT_COLLECTION_NAME, vector, limit)
