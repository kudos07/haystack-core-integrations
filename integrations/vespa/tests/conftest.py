# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S607

import os
import subprocess
import time

import pytest

from haystack_integrations.document_stores.vespa import VespaDocumentStore


@pytest.fixture(scope="session")
def deployed_vespa_app():
    """Deploy the bundled Vespa schema into the Dockerized Vespa instance (integration tests only)."""
    if os.environ.get("VESPA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("Set VESPA_RUN_INTEGRATION_TESTS=1 and start Vespa with docker compose.")

    subprocess.run(
        [
            "docker",
            "exec",
            "vespa",
            "bash",
            "-lc",
            "/opt/vespa/bin/vespa-deploy prepare /vespa_app && /opt/vespa/bin/vespa-deploy activate",
        ],
        check=True,
    )
    subprocess.run(
        ["docker", "exec", "vespa", "bash", "-lc", "/opt/vespa/bin/vespa-status deploy --wait 300"],
        check=True,
    )


def wait_until_documents_count(document_store, expected_count: int, *, deadline_s: float = 90) -> None:
    """Poll until Vespa search visibility matches ``expected_count`` (best-effort for CI)."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if document_store.count_documents() == expected_count:
            return
        time.sleep(0.5)
    msg = f"Timed out waiting for {expected_count} documents to be visible in Vespa."
    raise AssertionError(msg)


@pytest.fixture
def document_store(deployed_vespa_app, request):  # noqa: ARG001
    """Shared populated Vespa store for Haystack integration tests (see DocumentStoreBaseTests)."""
    _metadata_fields = [
        "category",
        "author",
        "name",
        "page",
        "chapter",
        "number",
        "date",
        "no_embedding",
        "year",
        "status",
        "updated",
        "extra_field",
        "featured",
        "priority",
        "rating",
        "age",
    ]
    store = VespaDocumentStore(
        url=os.environ.get("VESPA_URL", "http://localhost"),
        schema="doc",
        namespace="doc",
        content_field="content",
        embedding_field="embedding",
        metadata_fields=_metadata_fields,
    )
    store.delete_all_documents()
    yield store
    store.delete_all_documents()
