from haystack.document_stores.errors import DocumentStoreError


class VespaDocumentStoreError(DocumentStoreError):
    """Base exception for Vespa document store errors."""


class VespaDocumentStoreFilterError(VespaDocumentStoreError):
    """Raised when Haystack filters cannot be translated to Vespa YQL."""


class VespaDocumentStoreConfigError(VespaDocumentStoreError):
    """Raised when the Vespa document store configuration is invalid."""
