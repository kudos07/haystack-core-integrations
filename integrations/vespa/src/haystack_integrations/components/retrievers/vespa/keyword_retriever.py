from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
try:
    from haystack.document_stores.types import FilterPolicy, apply_filter_policy
except ImportError:  # pragma: no cover
    from haystack.document_stores.types.filter_policy import FilterPolicy, apply_filter_policy

from haystack_integrations.document_stores.vespa import VespaDocumentStore


@component
class VespaKeywordRetriever:
    """Retrieve documents from Vespa using lexical search."""

    def __init__(
        self,
        *,
        document_store: VespaDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        ranking: str | None = None,
    ) -> None:
        """
        Create a Vespa keyword retriever.

        :param document_store: Vespa document store instance.
        :param filters: Static retriever filters.
        :param top_k: Default number of documents to retrieve.
        :param filter_policy: Runtime filter merge policy.
        :param ranking: Optional Vespa ranking profile.
        """
        if not isinstance(document_store, VespaDocumentStore):
            msg = "document_store must be an instance of VespaDocumentStore"
            raise TypeError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )
        self._ranking = ranking

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
            filter_policy=self._filter_policy.value,
            ranking=self._ranking,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VespaKeywordRetriever":
        """
        Deserialize the retriever from a dictionary.

        :param data: Serialized retriever data.
        :returns: Deserialized retriever.
        """
        data["init_parameters"]["document_store"] = VespaDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(data["init_parameters"]["filter_policy"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from Vespa.

        :param query: Query text.
        :param filters: Optional runtime filters.
        :param top_k: Optional runtime `top_k`.
        :returns: Retrieved documents.
        """
        applied_filters = apply_filter_policy(self._filter_policy, self._filters, filters or {})
        documents = self._document_store._bm25_retrieval(
            query=query,
            filters=applied_filters or None,
            top_k=top_k or self._top_k,
            ranking=self._ranking,
        )
        return {"documents": documents}
