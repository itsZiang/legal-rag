import json
import logging
import os
import time

import numpy as np
from configs import DEFAULT_COLLECTION_NAME
from vectorize import add_vector
from bm25.sparse_text_embedding import SparseTextEmbedding


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "final_chunk")


def read_embeddings_file(file_path):
    """
    Read embeddings from JSON file and return the data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.info(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def process_embeddings(data):
    """
    Process the embeddings data and extract title, content, and vector values
    """
    ids = []
    titles = []
    contents = []
    vectors = []
    bm25_embedding_model = SparseTextEmbedding("davicn81/bm25")
    documents = []

    for item in data:
        ids.append(item.get("id"))
        titles.append(item.get("title"))
        contents.append(item.get("content"))
        vector = item.get("vector")
        # Convert to numpy array if you need to perform vector operations
        vector_np = np.array(vector) if vector else None
        vectors.append(vector_np)
        documents.append(item.get("title") + " " + item.get("content"))

    bm25_embeddings = list(bm25_embedding_model.embed(doc for doc in documents))

    return ids, titles, contents, vectors, bm25_embeddings


def indexing():
    file_path = os.path.join(DATA_DIR, "embeddings_output.json")
    data = read_embeddings_file(file_path)

    if data:
        ids, titles, contents, vectors, bm25_embeddings = process_embeddings(data)

        # Index each document
        for i in range(len(titles)):
            title = titles[i]
            content = contents[i]
            vector = vectors[i]
            bm25_embedding = bm25_embeddings[i]

            # Create a unique ID for each document
            doc_id = ids[i]

            # Index the document
            add_vector_status = add_vector(
                collection_name=DEFAULT_COLLECTION_NAME,
                vectors={
                    doc_id: {
                        "vector": {
                            "Greenode-Embedding-Large-VN-V1": vector,
                            "bm25": bm25_embedding.as_object(),
                        },
                        "payload": {"title": title, "content": content},
                    }
                },
            )

            logger.info(f"Indexed document {i}: {add_vector_status}")
    
    return True
