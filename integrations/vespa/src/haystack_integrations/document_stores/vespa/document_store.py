from __future__ import annotations

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from vespa.application import Vespa

from .errors import VespaDocumentStoreConfigError, VespaDocumentStoreError
from .filters import _normalize_filters

logger = logging.getLogger(__name__)

DEFAULT_QUERY_LIMIT = 400
DEFAULT_BULK_BATCH_SIZE = 100
HTTP_NOT_FOUND = 404


class VespaDocumentStore:
    """
    Document store backed by an existing [Vespa](https://vespa.ai/) application.
    """

    def __init__(
        self,
        *,
        url: Secret = Secret.from_env_var("VESPA_URL"),
        port: int = 8080,
        content_cluster_name: str = "content",
        schema: str = "doc",
        namespace: str | None = None,
        groupname: str | None = None,
        content_field: str = "content",
        embedding_field: str = "embedding",
        id_field: str = "id",
        metadata_fields: list[str] | None = None,
        query_limit: int = DEFAULT_QUERY_LIMIT,
    ) -> None:
        """
        Create a new Vespa document store.

        :param url: Vespa endpoint base URL.
        :param port: Vespa HTTP port.
        :param content_cluster_name: Vespa content cluster name.
        :param schema: Vespa schema name to read from and write to.
        :param namespace: Vespa namespace. Defaults to the schema name when omitted.
        :param groupname: Optional Vespa group name.
        :param content_field: Vespa field containing the document text.
        :param embedding_field: Vespa field containing the dense embedding.
        :param id_field: Optional Vespa field containing the document id in query responses.
            Vespa document IDs are always written via `data_id`. If this field is missing in the
            schema or summaries, the integration falls back to parsing the Vespa document path.
        :param metadata_fields: Optional allowlist of metadata fields to feed and return.
        :param query_limit: Maximum number of documents returned by bulk queries. Defaults to 400 to
            stay within Vespa's common query hit limit unless explicitly overridden.
        """
        self._url = url
        self._port = port
        self._content_cluster_name = content_cluster_name
        self._schema = schema
        self._namespace = namespace or schema
        self._groupname = groupname
        self._content_field = content_field
        self._embedding_field = embedding_field
        self._id_field = id_field
        self._metadata_fields = metadata_fields or []
        self._query_limit = query_limit
        self._app: Any | None = None

    @property
    def app(self) -> Any:
        """Return the underlying `pyvespa` client."""
        if self._app is None:
            resolved_url = self._url.resolve_value()
            if not resolved_url:
                msg = "A Vespa URL is required to initialize the document store"
                raise VespaDocumentStoreConfigError(msg)
            self._app = Vespa(url=resolved_url, port=self._port)
        return self._app

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the document store to a dictionary.

        :returns: Serialized document store data.
        """
        return default_to_dict(
            self,
            url=self._url.to_dict(),
            port=self._port,
            content_cluster_name=self._content_cluster_name,
            schema=self._schema,
            namespace=self._namespace,
            groupname=self._groupname,
            content_field=self._content_field,
            embedding_field=self._embedding_field,
            id_field=self._id_field,
            metadata_fields=self._metadata_fields,
            query_limit=self._query_limit,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VespaDocumentStore:
        """
        Deserialize the document store from a dictionary.

        :param data: Serialized document store data.
        :returns: Deserialized document store.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["url"])
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Return the total number of documents in Vespa.

        :returns: Document count.
        """
        response = self._query(yql=self._build_yql(where="true", limit=0), hits=0)
        return int(response.get("root", {}).get("fields", {}).get("totalCount", 0))

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Return the number of documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :returns: Count of matching documents.
        """
        where = _normalize_filters(filters, content_field=self._content_field)
        response = self._query(yql=self._build_yql(where=where, limit=0), hits=0)
        return int(response.get("root", {}).get("fields", {}).get("totalCount", 0))

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Write documents to Vespa.

        :param documents: Documents to store.
        :param policy: Duplicate handling policy.
        :returns: Number of documents written.
        """
        written = 0
        for document in documents:
            if policy == DuplicatePolicy.FAIL and self._document_exists(document.id):
                msg = f"Document with id '{document.id}' already exists in Vespa"
                raise VespaDocumentStoreError(msg)
            if policy == DuplicatePolicy.SKIP and self._document_exists(document.id):
                continue

            response = self.app.feed_data_point(
                schema=self._schema,
                namespace=self._namespace,
                groupname=self._groupname,
                data_id=document.id,
                fields=self._document_to_vespa_fields(document),
            )
            self._ensure_success(response, f"Failed to write document '{document.id}' to Vespa")
            written += 1
        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by id.

        :param document_ids: Document ids to delete.
        """
        for document_id in document_ids:
            response = self.app.delete_data(
                schema=self._schema,
                namespace=self._namespace,
                groupname=self._groupname,
                data_id=document_id,
            )
            status_code = getattr(response, "status_code", 200)
            if status_code not in {200, 202, 204, 404}:
                self._ensure_success(response, f"Failed to delete document '{document_id}' from Vespa")

    def delete_all_documents(self) -> None:
        """Delete all documents in the configured Vespa schema."""
        documents = self._collect_matching_documents()
        if not documents:
            return
        self.delete_documents([document.id for document in documents])

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Delete all documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :returns: Number of deleted documents.
        """
        documents = self._collect_matching_documents(filters=filters)
        self.delete_documents([document.id for document in documents])
        return len(documents)

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Update metadata fields for documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :param meta: Metadata values to merge into the matched documents.
        :returns: Number of updated documents.
        """
        documents = self._collect_matching_documents(filters=filters)
        updated = 0
        for document in documents:
            response = self.app.update_data(
                schema=self._schema,
                namespace=self._namespace,
                groupname=self._groupname,
                data_id=document.id,
                fields=dict(meta),
                create=False,
            )
            self._ensure_success(response, f"Failed to update document '{document.id}' in Vespa")
            updated += 1
        return updated

    def get_documents_by_id(self, document_ids: list[str]) -> list[Document]:
        """
        Retrieve documents by their ids.

        :param document_ids: Document ids to fetch.
        :returns: Matching documents.
        """
        documents: list[Document] = []
        for document_id in document_ids:
            response = self.app.get_data(
                schema=self._schema,
                namespace=self._namespace,
                groupname=self._groupname,
                data_id=document_id,
                raise_on_not_found=False,
            )
            status_code = getattr(response, "status_code", 200)
            if status_code == HTTP_NOT_FOUND:
                continue
            self._ensure_success(response, f"Failed to retrieve document '{document_id}' from Vespa")
            payload = response.get_json() if hasattr(response, "get_json") else getattr(response, "json", {})
            fields = payload.get("fields", {})
            documents.append(self._fields_to_document(fields, score=None, fallback_id=document_id))
        return documents

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Retrieve documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :returns: Matching documents.
        """
        where = _normalize_filters(filters, content_field=self._content_field) if filters else "true"
        return self._query_documents(where=where, top_k=self._query_limit)

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Return best-effort metadata field information based on configured fields.

        :returns: Field metadata information.
        """
        info = {"content": {"type": "text"}}
        for field in self._metadata_fields:
            info[field] = {"type": "keyword"}
        return info

    def _bm25_retrieval(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        ranking: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents using Vespa lexical search.

        :param query: Query text.
        :param top_k: Maximum number of documents to return.
        :param filters: Optional Haystack metadata filters.
        :param ranking: Optional Vespa ranking profile.
        :returns: Retrieved documents.
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        where = _normalize_filters(filters, content_field=self._content_field) if filters else "true"
        clauses = [where, "userQuery()"] if where != "true" else ["userQuery()"]
        return self._query_documents(where=" and ".join(clauses), top_k=top_k, query=query, ranking=ranking)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        ranking: str | None = None,
        query_tensor_name: str = "query_embedding",
        target_hits: int | None = None,
    ) -> list[Document]:
        """
        Retrieve documents using Vespa nearest-neighbor search.

        :param query_embedding: Query embedding vector.
        :param top_k: Maximum number of documents to return.
        :param filters: Optional Haystack metadata filters.
        :param ranking: Optional Vespa ranking profile.
        :param query_tensor_name: Query tensor name referenced in YQL.
        :param target_hits: Optional Vespa nearest-neighbor `targetHits` value.
        :returns: Retrieved documents.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        where = _normalize_filters(filters, content_field=self._content_field) if filters else "true"
        nearest = (
            f"{{targetHits:{target_hits}}}nearestNeighbor({self._embedding_field}, {query_tensor_name})"
            if target_hits
            else f"nearestNeighbor({self._embedding_field}, {query_tensor_name})"
        )
        clauses = [where, nearest] if where != "true" else [nearest]
        body = {f"input.query({query_tensor_name})": query_embedding}
        return self._query_documents(where=" and ".join(clauses), top_k=top_k, ranking=ranking, body=body)

    def _query_documents(
        self,
        *,
        where: str,
        top_k: int,
        offset: int = 0,
        query: str | None = None,
        ranking: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> list[Document]:
        yql = self._build_yql(where=where, limit=top_k, offset=offset)
        payload = self._query(yql=yql, query=query, hits=top_k, offset=offset, ranking=ranking, body=body or {})
        hits = payload.get("root", {}).get("children", [])
        return [self._hit_to_document(hit) for hit in hits]

    def _query(
        self,
        *,
        yql: str,
        query: str | None = None,
        hits: int | None = None,
        offset: int | None = None,
        ranking: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {"yql": yql}
        if query is not None:
            request_body["query"] = query
        if hits is not None:
            request_body["hits"] = hits
        if offset is not None:
            request_body["offset"] = offset
        if ranking is not None:
            request_body["ranking"] = ranking
        if body:
            request_body.update(body)

        response = self.app.query(body=request_body)
        self._ensure_success(response, "Failed to query Vespa")
        if hasattr(response, "get_json"):
            return response.get_json()
        return getattr(response, "json", {})

    def _document_exists(self, document_id: str) -> bool:
        response = self.app.get_data(
            schema=self._schema,
            namespace=self._namespace,
            groupname=self._groupname,
            data_id=document_id,
            raise_on_not_found=False,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code == HTTP_NOT_FOUND:
            return False
        self._ensure_success(response, f"Failed to check whether document '{document_id}' exists in Vespa")
        return True

    def _ensure_success(self, response: Any, message: str) -> None:
        if hasattr(response, "is_successful") and response.is_successful():
            return
        status_code = getattr(response, "status_code", "unknown")
        payload = response.get_json() if hasattr(response, "get_json") else getattr(response, "json", None)
        error_message = f"{message}. Status code: {status_code}. Response: {payload}"
        raise VespaDocumentStoreError(error_message)

    def _build_yql(self, *, where: str, limit: int, offset: int = 0) -> str:
        return f"select * from sources {self._schema} where {where} limit {limit} offset {offset}"  # noqa: S608

    def _document_to_vespa_fields(self, document: Document) -> dict[str, Any]:
        doc_dict = document.to_dict(flatten=False)
        fields: dict[str, Any] = {}

        if document.content is not None:
            fields[self._content_field] = document.content
        if document.embedding is not None:
            fields[self._embedding_field] = document.embedding

        metadata = doc_dict.get("meta", {}) or {}
        if self._metadata_fields:
            for key in self._metadata_fields:
                if key in metadata:
                    fields[key] = metadata[key]
        else:
            fields.update(metadata)
        return fields

    def _collect_matching_documents(
        self, *, filters: dict[str, Any] | None = None, batch_size: int = DEFAULT_BULK_BATCH_SIZE
    ) -> list[Document]:
        where = _normalize_filters(filters, content_field=self._content_field) if filters else "true"
        documents: list[Document] = []
        offset = 0
        effective_batch_size = min(batch_size, self._query_limit)

        while True:
            batch = self._query_documents(where=where, top_k=effective_batch_size, offset=offset)
            if not batch:
                break
            documents.extend(batch)
            if len(batch) < effective_batch_size:
                break
            offset += effective_batch_size

        return documents

    def _fields_to_document(self, fields: dict[str, Any], *, score: float | None, fallback_id: str | None) -> Document:
        document_id = fields.get(self._id_field) or fallback_id
        if document_id is None:
            msg = "Vespa response does not contain a document id"
            raise VespaDocumentStoreError(msg)

        meta = {
            key: value
            for key, value in fields.items()
            if key not in {self._id_field, self._content_field, self._embedding_field}
        }
        if self._metadata_fields:
            meta = {key: value for key, value in meta.items() if key in self._metadata_fields}

        return Document(
            id=str(document_id),
            content=fields.get(self._content_field),
            embedding=fields.get(self._embedding_field),
            meta=meta,
            score=score,
        )

    def _hit_to_document(self, hit: dict[str, Any]) -> Document:
        fields = hit.get("fields", {})
        fallback_id = self._extract_document_id(hit.get("id"))
        return self._fields_to_document(fields, score=hit.get("relevance"), fallback_id=fallback_id)

    @staticmethod
    def _extract_document_id(raw_id: str | None) -> str | None:
        if raw_id is None:
            return None
        if "::" in raw_id:
            return raw_id.rsplit("::", maxsplit=1)[-1]
        return raw_id.rsplit("/", maxsplit=1)[-1]
