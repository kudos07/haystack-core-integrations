from copy import deepcopy
from unittest.mock import Mock

from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_integrations.components.retrievers.vespa import VespaEmbeddingRetriever, VespaKeywordRetriever
from haystack_integrations.document_stores.vespa import VespaDocumentStore


def _document_store():
    return VespaDocumentStore(url="http://localhost", schema="docs", namespace="docs")


def test_keyword_retriever_run():
    document_store = _document_store()
    document_store._bm25_retrieval = Mock(return_value=[Document(id="1", content="hello")])

    retriever = VespaKeywordRetriever(
        document_store=document_store,
        filters={"field": "meta.category", "operator": "==", "value": "news"},
        ranking="bm25",
    )
    result = retriever.run("hello")

    assert result["documents"][0].id == "1"
    document_store._bm25_retrieval.assert_called_once()


def test_keyword_retriever_defaults_to_bm25_ranking():
    document_store = _document_store()
    document_store._bm25_retrieval = Mock(return_value=[Document(id="1", content="hello")])

    retriever = VespaKeywordRetriever(document_store=document_store)
    retriever.run("hello")

    _, kwargs = document_store._bm25_retrieval.call_args
    assert kwargs["ranking"] == "bm25"


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


def test_embedding_retriever_defaults_to_semantic_ranking():
    document_store = _document_store()
    document_store._embedding_retrieval = Mock(return_value=[Document(id="1", content="hello")])

    retriever = VespaEmbeddingRetriever(document_store=document_store)
    retriever.run([0.1, 0.2, 0.3])

    _, kwargs = document_store._embedding_retrieval.call_args
    assert kwargs["ranking"] == "semantic"


def test_keyword_retriever_component_serialization_roundtrip():
    document_store = VespaDocumentStore(url="http://localhost", schema="docs", namespace="docs")
    original = VespaKeywordRetriever(
        document_store=document_store,
        filters={"field": "meta.category", "operator": "==", "value": "news"},
        top_k=7,
        ranking="bm25",
    )
    data = component_to_dict(original, "retriever")
    expected = deepcopy(data)
    restored = component_from_dict(VespaKeywordRetriever, data, "retriever")
    assert component_to_dict(restored, "retriever") == expected


def test_embedding_retriever_component_serialization_roundtrip():
    document_store = VespaDocumentStore(url="http://localhost", schema="docs", namespace="docs")
    original = VespaEmbeddingRetriever(
        document_store=document_store,
        filters={"field": "meta.category", "operator": "==", "value": "news"},
        top_k=7,
        ranking="semantic",
        query_tensor_name="query_embedding",
        target_hits=40,
    )
    data = component_to_dict(original, "retriever")
    expected = deepcopy(data)
    restored = component_from_dict(VespaEmbeddingRetriever, data, "retriever")
    assert component_to_dict(restored, "retriever") == expected
