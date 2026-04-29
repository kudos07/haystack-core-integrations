from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict

from haystack_integrations.document_stores.vespa import VespaDocumentStore


@component
class VespaEmbeddingRetriever:
    """Retrieve documents from Vespa using dense vector similarity."""

    def __init__(
        self,
        *,
        document_store: VespaDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        ranking: str | None = None,
        query_tensor_name: str = "query_embedding",
        target_hits: int | None = None,
    ) -> None:
        """
        Create a Vespa embedding retriever.

        :param document_store: Vespa document store instance.
        :param filters: Static retriever filters.
        :param top_k: Default number of documents to retrieve.
        :param ranking: Optional Vespa ranking profile.
        :param query_tensor_name: Query tensor name referenced in Vespa YQL.
        :param target_hits: Optional Vespa nearest-neighbor `targetHits` value.
        """
        if not isinstance(document_store, VespaDocumentStore):
            msg = "document_store must be an instance of VespaDocumentStore"
            raise TypeError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._ranking = ranking
        self._query_tensor_name = query_tensor_name
        self._target_hits = target_hits

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the retriever to a dictionary.

        :returns: Serialized retriever data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            top_k=self._top_k,
            ranking=self._ranking,
            query_tensor_name=self._query_tensor_name,
            target_hits=self._target_hits,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VespaEmbeddingRetriever":
        """
        Deserialize the retriever from a dictionary.

        :param data: Serialized retriever data.
        :returns: Deserialized retriever.
        """
        data["init_parameters"]["document_store"] = VespaDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from Vespa.

        :param query_embedding: Dense query embedding.
        :param filters: Optional runtime filters.
        :param top_k: Optional runtime `top_k`.
        :returns: Retrieved documents.
        """
        applied_filters = filters or self._filters
        documents = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=applied_filters or None,
            top_k=top_k or self._top_k,
            ranking=self._ranking,
            query_tensor_name=self._query_tensor_name,
            target_hits=self._target_hits,
        )
        return {"documents": documents}
