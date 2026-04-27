from unittest.mock import Mock

from haystack import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils import Secret

from haystack_integrations.components.retrievers.vespa import VespaEmbeddingRetriever, VespaKeywordRetriever
from haystack_integrations.document_stores.vespa import VespaDocumentStore


def _document_store():
    return VespaDocumentStore(url=Secret.from_token("http://localhost"), schema="docs", namespace="docs")


def test_keyword_retriever_run():
    document_store = _document_store()
    document_store._bm25_retrieval = Mock(return_value=[Document(id="1", content="hello")])

    retriever = VespaKeywordRetriever(
        document_store=document_store,
        filters={"field": "meta.category", "operator": "==", "value": "news"},
        filter_policy=FilterPolicy.REPLACE,
        ranking="bm25",
    )
    result = retriever.run("hello")

    assert result["documents"][0].id == "1"
    document_store._bm25_retrieval.assert_called_once()


def test_embedding_retriever_run():
    document_store = _document_store()
    document_store._embedding_retrieval = Mock(return_value=[Document(id="1", content="hello")])

    retriever = VespaEmbeddingRetriever(
        document_store=document_store,
        query_tensor_name="q",
        target_hits=25,
    )
    result = retriever.run([0.1, 0.2, 0.3], top_k=3)

    assert result["documents"][0].id == "1"
    _, kwargs = document_store._embedding_retrieval.call_args
    assert kwargs["query_tensor_name"] == "q"
    assert kwargs["target_hits"] == 25
    assert kwargs["top_k"] == 3
