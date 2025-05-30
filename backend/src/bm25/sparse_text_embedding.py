from typing import Any, Iterable, Optional, Type, Union
from dataclasses import asdict

from bm25.bm25 import Bm25
from bm25.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from bm25.model_description import SparseModelDescription


class SparseTextEmbedding(SparseTextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[Type[SparseTextEmbeddingBase]] = [Bm25]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "prithvida/SPLADE_PP_en_v1",
                        "vocab_size": 30522,
                        "description": "Independent Implementation of SPLADE++ Model for English",
                        "license": "apache-2.0",
                        "size_in_GB": 0.532,
                        "sources": {
                            "hf": "qdrant/SPLADE_PP_en_v1",
                        },
                    }
                ]
                ```
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        result: list[SparseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in SparseTextEmbedding."
            "Please check the supported models using `SparseTextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        yield from self.model.embed(documents, batch_size, parallel, **kwargs)

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """
        yield from self.model.query_embed(query, **kwargs)
