from rerank import rerank_documents_v2
from bm25.sparse_text_embedding import SparseTextEmbedding
from brain import get_embedding
from qdrant_client import models
from configs import DEFAULT_COLLECTION_NAME
from qdrant_client import QdrantClient

# Khởi tạo Qdrant client và mô hình BM25
client = QdrantClient(url="http://qdrant-db:6333")
bm25_embedding_model = SparseTextEmbedding("davicn81/bm25")


def hybrid_search(query: str, limit=20, top_n=20):
    # 1. Lấy embedding dense
    dense_vector = get_embedding(query)

    # 2. Lấy embedding sparse từ mô hình BM25
    try:
        # Giả sử bm25_embedding_model.query_embed trả về một iterable
        bm25_embedding = next(iter(bm25_embedding_model.query_embed(query))).as_object()
    except StopIteration:
        raise ValueError("Không tìm thấy BM25 embedding cho câu truy vấn.")

    # 3. Tạo SparseVector cho BM25 embedding
    sparse_vector = models.SparseVector(
        indices=bm25_embedding[
            "indices"
        ],  # Chắc chắn rằng bm25_embedding có trường indices
        values=bm25_embedding[
            "values"
        ],  # Chắc chắn rằng bm25_embedding có trường values
    )

    # 4. Truy vấn Qdrant bằng vector dense
    dense_results = client.search(
        collection_name=DEFAULT_COLLECTION_NAME,
        query_vector=models.NamedVector(
            name="Greenode-Embedding-Large-VN-V1", vector=dense_vector
        ),
        with_payload=True,
        limit=limit,
    )

    # 5. Truy vấn Qdrant bằng vector sparse (BM25)
    bm25_results = client.search(
        collection_name=DEFAULT_COLLECTION_NAME,
        query_vector=models.NamedSparseVector(name="bm25", vector=sparse_vector),
        with_payload=True,
        limit=limit,
    )

    # 6. Kết hợp kết quả và loại bỏ trùng ID
    combined_results = dense_results + bm25_results
    unique_results = {point.id: point for point in combined_results}.values()
    unique_results = list(unique_results)
    # 7. Rerank kết quả
    reranked_results = rerank_documents_v2(
        docs=[
            {
                "id": it.id,
                "title": it.payload.get("title", ""),
                "content": it.payload.get("content", ""),
            }
            for it in unique_results
        ],
        query=query,
        top_n=top_n,
    )

    return reranked_results
