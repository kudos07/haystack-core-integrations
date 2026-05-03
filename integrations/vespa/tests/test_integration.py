# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from haystack import Document

from haystack_integrations.components.retrievers.vespa import VespaEmbeddingRetriever, VespaKeywordRetriever

EXPECTED_DOCUMENT_COUNT = 3


def _wait_until_documents_are_visible(document_store, expected_count: int) -> None:
    deadline_s = 90.0
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if document_store.count_documents() == expected_count:
            return
        time.sleep(0.5)
    msg = f"Expected {expected_count} documents to become visible in Vespa"
    raise AssertionError(msg)


@pytest.mark.integration
def test_vespa_keyword_and_embedding_retrieval(document_store):
    written = document_store.write_documents(
        [
            Document(
                id="1",
                content="Haystack integrates with Vespa for search.",
                embedding=[1.0, 0.0, 0.0],
                meta={"category": "docs", "author": "deepset"},
            ),
            Document(
                id="2",
                content="Vespa supports lexical and vector retrieval.",
                embedding=[0.0, 1.0, 0.0],
                meta={"category": "docs", "author": "vespa"},
            ),
            Document(
                id="3",
                content="This note is about something else entirely.",
                embedding=[0.0, 0.0, 1.0],
                meta={"category": "misc", "author": "someone"},
            ),
        ]
    )

    assert written == EXPECTED_DOCUMENT_COUNT
    _wait_until_documents_are_visible(document_store, EXPECTED_DOCUMENT_COUNT)

    keyword_retriever = VespaKeywordRetriever(
        document_store=document_store,
        top_k=2,
        filters={"field": "meta.category", "operator": "==", "value": "docs"},
    )
    keyword_result = keyword_retriever.run(query="vector retrieval")

    assert keyword_result["documents"]
    assert keyword_result["documents"][0].id == "2"

    embedding_retriever = VespaEmbeddingRetriever(document_store=document_store, top_k=1)
    embedding_result = embedding_retriever.run(query_embedding=[1.0, 0.0, 0.0])

    assert embedding_result["documents"]
    assert embedding_result["documents"][0].id == "1"
