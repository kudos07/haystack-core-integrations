# vespa-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/vespa-haystack.svg)](https://pypi.org/project/vespa-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vespa-haystack.svg)](https://pypi.org/project/vespa-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/vespa/CHANGELOG.md)

---

`vespa-haystack` provides a Haystack `DocumentStore` plus keyword and embedding retrievers for
[Vespa](https://vespa.ai/).

This integration assumes you already have a Vespa application and schema running. The document store
connects to that existing setup and lets you write documents and query them from Haystack pipelines.

## Examples

- [Keyword retrieval example](examples/keyword_retrieval.py)
- [Embedding retrieval example](examples/embedding_retrieval.py)

## Local Smoke Test

To verify the integration against a real local Vespa instance, start Docker Desktop and run:

```bash
hatch run python scripts/local_keyword_smoke_test.py
```

This deploys a minimal Vespa application locally, writes three documents, runs a direct filter query,
and checks keyword retrieval through `VespaKeywordRetriever`.

## Notes

- Set `VESPA_URL` to your Vespa endpoint before running the examples.
- Make sure your Vespa schema field names match the ones you pass into `VespaDocumentStore`.
- Vespa document IDs are written through the Vespa document path (`data_id`). The optional `id_field`
  is only used when a query response also exposes an explicit id field.
- For embedding retrieval, your Vespa schema must already include a tensor field and a ranking profile
  compatible with nearest-neighbor search. The example assumes a ranking profile named `semantic`.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
