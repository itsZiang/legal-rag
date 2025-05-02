import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Modifier,
    SparseVectorParams,
    NamedVector,
    NamedSparseVector,
)

logger = logging.getLogger(__name__)
client = QdrantClient(url="http://qdrant-db:6333")


def create_collection(name):
    return client.create_collection(
        collection_name=name,
        vectors_config={
            "Greenode-Embedding-Large-VN-V1": VectorParams(
                size=1024, distance=Distance.COSINE
            )
        },
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )


def add_vector(collection_name, vectors={}):
    points = [
        PointStruct(id=k, vector=v["vector"], payload=v["payload"])
        for k, v in vectors.items()
    ]
    return client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )


def search_vector(collection_name, vector, limit=2):
    res = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name="Greenode-Embedding-Large-VN-V1", vector=vector),
        limit=limit,
    )
    payloads = [x.payload for x in res]
    return payloads


def search_vector_ids(collection_name, vector, limit=2):
    res = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name="Greenode-Embedding-Large-VN-V1", vector=vector),
        limit=limit,
    )
    return [x.id for x in res]


def search_vector_ids_payloads(collection_name, vector, limit=2):
    res = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name="Greenode-Embedding-Large-VN-V1", vector=vector),
        limit=limit,
    )
    ids = [x.id for x in res]
    payloads = [x.payload for x in res]
    output = [{"id": id, "payload": payload} for id, payload in zip(ids, payloads)]
    return output
