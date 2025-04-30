from rerank import rerank_documents_v2
from brain import get_embedding
from vectorize import search_vector_ids, search_vector_ids_payloads
from configs import DEFAULT_COLLECTION_NAME


def search_documents_ids(query, limit=5):
    vector = get_embedding(query)
    search_ids = search_vector_ids(DEFAULT_COLLECTION_NAME, vector, limit)
    return search_ids

def search_documents_ids_rerank(query, limit=5, top_n=5):
    """
    1) retrieve (id, payload) pairs via Qdrant
    2) build docs with `title` + `content`
    3) call the reranker
    4) return only the ids in new order
    """
    vector = get_embedding(query)
    items = search_vector_ids_payloads(DEFAULT_COLLECTION_NAME, vector, limit)
    # payload must contain 'title' & 'content'
    docs = [
        {"id": it["id"],
         "title": it["payload"].get("title", ""),
         "content": it["payload"].get("content", "")}
        for it in items
    ]
    ranked = rerank_documents_v2(docs, query, top_n)
    return [doc["id"] for doc in ranked]
