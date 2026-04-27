from __future__ import annotations

import logging

from haystack.dataclasses import Document as HaystackDocument
from haystack.utils import Secret
from vespa.deployment import VespaDocker
from vespa.package import ApplicationPackage, Document, Field, FieldSet, RankProfile, Schema

from haystack_integrations.components.retrievers.vespa import VespaKeywordRetriever
from haystack_integrations.document_stores.vespa import VespaDocumentStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
EXPECTED_DOCUMENT_COUNT = 3
EXPECTED_FILTERED_DOCUMENT_COUNT = 2


def _build_application_package() -> ApplicationPackage:
    schema = Schema(
        name="doc",
        document=Document(
            fields=[
                Field(name="content", type="string", indexing=["index", "summary"], index="enable-bm25"),
                Field(name="category", type="string", indexing=["attribute", "summary"]),
                Field(name="author", type="string", indexing=["attribute", "summary"]),
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["content"])],
        rank_profiles=[RankProfile(name="bm25", first_phase="bm25(content)")],
    )
    return ApplicationPackage(name="vespasmoke", schema=[schema])


def main() -> None:
    """Deploy a local Vespa app and verify the keyword integration end to end."""
    logger.info("Deploying a minimal Vespa application locally")
    docker = VespaDocker(url="http://localhost", port=8080, cfgsrv_port=19071)
    app = docker.deploy(_build_application_package(), debug=False)
    app.wait_for_application_up(max_wait=300)

    document_store = VespaDocumentStore(
        url=Secret.from_token("http://localhost"),
        schema="doc",
        namespace="doc",
        content_field="content",
        metadata_fields=["category", "author"],
    )

    logger.info("Resetting the Vespa schema and indexing sample documents")
    document_store.delete_all_documents()
    written = document_store.write_documents(
        [
            HaystackDocument(
                id="1",
                content="Haystack integrates with Vespa for search.",
                meta={"category": "docs", "author": "deepset"},
            ),
            HaystackDocument(
                id="2",
                content="Vespa supports lexical and vector retrieval.",
                meta={"category": "docs", "author": "vespa"},
            ),
            HaystackDocument(
                id="3",
                content="This note is about something else entirely.",
                meta={"category": "misc", "author": "someone"},
            ),
        ]
    )
    if written != EXPECTED_DOCUMENT_COUNT:
        msg = f"Expected to write {EXPECTED_DOCUMENT_COUNT} documents, wrote {written}"
        raise RuntimeError(msg)

    count = document_store.count_documents()
    if count != EXPECTED_DOCUMENT_COUNT:
        msg = f"Expected {EXPECTED_DOCUMENT_COUNT} documents in Vespa, found {count}"
        raise RuntimeError(msg)

    filtered = document_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "docs"})
    if len(filtered) != EXPECTED_FILTERED_DOCUMENT_COUNT:
        msg = f"Expected {EXPECTED_FILTERED_DOCUMENT_COUNT} filtered documents, found {len(filtered)}"
        raise RuntimeError(msg)

    retriever = VespaKeywordRetriever(
        document_store=document_store,
        top_k=2,
        filters={"field": "meta.category", "operator": "==", "value": "docs"},
        ranking="bm25",
    )
    result = retriever.run(query="vector retrieval")
    documents = result["documents"]

    if not documents:
        msg = "Keyword retrieval returned no documents"
        raise RuntimeError(msg)

    top_document = documents[0]
    if top_document.id != "2":
        msg = f"Expected top document id '2', got '{top_document.id}'"
        raise RuntimeError(msg)

    logger.info("Smoke test passed")
    logger.info("Wrote %s documents", written)
    logger.info("Filtered docs: %s", [doc.id for doc in filtered])
    logger.info("Keyword retriever results: %s", [doc.id for doc in documents])


if __name__ == "__main__":
    main()
