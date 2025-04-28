import cohere
import requests
import numpy as np
import os
from time import time
from typing import List
from configs import (
    VAST_IP_ADDRESS_EMBED_RERANK,
    VAST_PORT_EMBED_RERANK,
)

# Set up your cohere client
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
DEFAULT_RANK_MODEL = "rerank-v3.5"
co = cohere.Client(COHERE_API_KEY)


def rerank_documents(docs, query, top_n=3, rank_model=DEFAULT_RANK_MODEL):
    """
    Rerank documents based on the query
    """
    process_docs = [doc["title"] + " " + doc["content"] for doc in docs]
    results = co.rerank(
        query=query, documents=process_docs, top_n=top_n, model=rank_model
    )
    for item in results.results:
        # print(f"Document Index: {item.index}")
        print(f"Rerank. Document: {docs[item.index]}")
        print(f"Relevance Score: {item.relevance_score:.5f}")

    ranked_docs = [docs[item.index] for item in results.results]

    return ranked_docs


def rerank_documents_v2(docs, query, top_n=3, rank_model=DEFAULT_RANK_MODEL):
    """
    Rerank documents based on the query
    """
    process_docs = [doc["title"] + " " + doc["content"] for doc in docs]
    response = requests.post(
        f"http://{VAST_IP_ADDRESS_EMBED_RERANK}:{VAST_PORT_EMBED_RERANK}/rerank",
        json={
            "query": query,
            "documents": process_docs,
            "top_n": top_n
        }
    )
    results = response.json()
    for item in results["results"]:
        # print(f"Document Index: {item.index}")
        print(f"Rerank. Document: {docs[item.index]}")
        print(f"Relevance Score: {item.relevance_score:.5f}")

    ranked_docs = [docs[item.index] for item in results.results]

    return ranked_docs
