import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VESPA_RUN_INTEGRATION_TESTS") != "1",
    reason="Set VESPA_RUN_INTEGRATION_TESTS=1 and provide a real Vespa setup to run integration tests.",
)
def test_vespa_integration_placeholder():
    """Placeholder integration test to keep CI collection stable until Vespa-backed tests are added."""
