import os
from unittest.mock import Mock

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.vespa import VespaDocumentStore
from haystack_integrations.document_stores.vespa.errors import VespaDocumentStoreError
from haystack_integrations.document_stores.vespa.filters import _normalize_filters


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def is_successful(self):
        return 200 <= self.status_code < 300

    def get_json(self):
        return self._payload


@pytest.fixture
def store():
    document_store = VespaDocumentStore(
        url=Secret.from_token("http://localhost"),
        schema="docs",
        namespace="docs",
        metadata_fields=["category", "author"],
    )
    document_store._app = Mock()
    return document_store


def test_to_dict_from_dict():
    os.environ["VESPA_TEST_URL"] = "http://localhost"
    document_store = VespaDocumentStore(
        url=Secret.from_env_var("VESPA_TEST_URL"),
        schema="docs",
        namespace="docs",
        metadata_fields=["category"],
    )

    restored = VespaDocumentStore.from_dict(document_store.to_dict())

    assert restored.to_dict() == document_store.to_dict()


def test_count_documents(store):
    store._app.query.return_value = DummyResponse({"root": {"fields": {"totalCount": 7}}})

    assert store.count_documents() == 7


def test_string_equality_filters_use_contains():
    yql_filter = _normalize_filters(
        {"field": "meta.category", "operator": "==", "value": "news"},
        content_field="content",
    )

    assert yql_filter == 'category contains "news"'


def test_write_documents(store):
    store._app.feed_data_point.return_value = DummyResponse({})

    written = store.write_documents(
        [Document(id="1", content="hello", embedding=[0.1, 0.2], meta={"category": "news", "ignored": "x"})]
    )

    assert written == 1
    _, kwargs = store._app.feed_data_point.call_args
    assert kwargs["schema"] == "docs"
    assert kwargs["data_id"] == "1"
    assert kwargs["fields"]["content"] == "hello"
    assert kwargs["fields"]["embedding"] == [0.1, 0.2]
    assert kwargs["fields"]["category"] == "news"
    assert "id" not in kwargs["fields"]
    assert "ignored" not in kwargs["fields"]


def test_write_documents_duplicate_skip(store):
    store._app.get_data.return_value = DummyResponse({"fields": {"id": "1"}})

    written = store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)

    assert written == 0
    store._app.feed_data_point.assert_not_called()


def test_write_documents_duplicate_check_surfaces_backend_error(store):
    store._app.get_data.return_value = DummyResponse({"message": "boom"}, status_code=500)

    with pytest.raises(VespaDocumentStoreError):
        store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)


def test_filter_documents(store):
    store._app.query.return_value = DummyResponse(
        {
            "root": {
                "children": [
                    {
                        "id": "id:docs:docs::1",
                        "relevance": 3.5,
                        "fields": {
                            "id": "1",
                            "content": "hello",
                            "embedding": [0.1, 0.2],
                            "category": "news",
                        },
                    }
                ]
            }
        }
    )

    documents = store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "news"})

    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].score == 3.5
    assert documents[0].meta == {"category": "news"}


def test_get_documents_by_id(store):
    store._app.get_data.return_value = DummyResponse({"fields": {"id": "1", "content": "hello", "author": "sam"}})

    documents = store.get_documents_by_id(["1"])

    assert [doc.id for doc in documents] == ["1"]
    assert documents[0].meta == {"author": "sam"}


def test_delete_by_filter(store):
    store._app.query.side_effect = [
        DummyResponse(
            {
                "root": {
                    "children": [
                        {"id": "id:docs:docs::1", "fields": {"content": "hello"}},
                        {"id": "id:docs:docs::2", "fields": {"content": "world"}},
                    ]
                }
            }
        ),
        DummyResponse({"root": {"children": []}}),
    ]
    store._app.delete_data.return_value = DummyResponse({})

    deleted = store.delete_by_filter(filters={"field": "meta.category", "operator": "==", "value": "news"})

    assert deleted == 2
    assert store._app.delete_data.call_count == 2


def test_delete_all_documents(store):
    store._app.query.side_effect = [
        DummyResponse({"root": {"children": [{"id": "id:docs:docs::1", "fields": {"content": "hello"}}]}}),
        DummyResponse({"root": {"children": []}}),
    ]
    store._app.delete_data.return_value = DummyResponse({})

    store.delete_all_documents()

    _, kwargs = store._app.delete_data.call_args
    assert kwargs["schema"] == "docs"
    assert kwargs["data_id"] == "1"
