from __future__ import annotations

import json
from typing import Any

from .errors import VespaDocumentStoreFilterError


def _format_yql_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _is_scalar_string_value(value: Any) -> bool:
    return isinstance(value, str)


def _normalize_field_name(field: str, *, content_field: str) -> str:
    if field == "content":
        return content_field
    if field.startswith("meta."):
        return field[5:]
    return field


def _convert_comparison(
    field: str,
    operator: str,
    value: Any,
    *,
    content_field: str,
) -> str:
    normalized_field = _normalize_field_name(field, content_field=content_field)

    if operator == "==":
        if _is_scalar_string_value(value):
            return f"{normalized_field} contains {_format_yql_value(value)}"
        return f"{normalized_field} = {_format_yql_value(value)}"
    if operator == "!=":
        if _is_scalar_string_value(value):
            return f"!( {normalized_field} contains {_format_yql_value(value)} )"
        return f"!( {normalized_field} = {_format_yql_value(value)} )"
    if operator == ">":
        return f"{normalized_field} > {_format_yql_value(value)}"
    if operator == ">=":
        return f"{normalized_field} >= {_format_yql_value(value)}"
    if operator == "<":
        return f"{normalized_field} < {_format_yql_value(value)}"
    if operator == "<=":
        return f"{normalized_field} <= {_format_yql_value(value)}"
    if operator == "in":
        if not isinstance(value, list):
            msg = "'in' filter values must be lists"
            raise VespaDocumentStoreFilterError(msg)
        values = ", ".join(_format_yql_value(item) for item in value)
        return f"{normalized_field} in ({values})"
    if operator == "not in":
        if not isinstance(value, list):
            msg = "'not in' filter values must be lists"
            raise VespaDocumentStoreFilterError(msg)
        values = ", ".join(_format_yql_value(item) for item in value)
        return f"!( {normalized_field} in ({values}) )"
    if operator == "contains":
        return f"{normalized_field} contains {_format_yql_value(value)}"
    if operator == "not contains":
        return f"!( {normalized_field} contains {_format_yql_value(value)} )"

    msg = f"Unsupported Vespa filter operator: {operator}"
    raise VespaDocumentStoreFilterError(msg)


def _normalize_filters(filters: dict[str, Any] | None, *, content_field: str) -> str:
    """
    Convert Haystack metadata filters into a Vespa YQL expression.

    :param filters: Haystack-style filters.
    :param content_field: Vespa field name used for document content.
    :returns: Vespa YQL expression without the enclosing `where`.
    """
    if not filters:
        return "true"

    operator = filters.get("operator")
    if operator in {"AND", "OR"}:
        conditions = filters.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            msg = f"{operator} filters must contain a non-empty 'conditions' list"
            raise VespaDocumentStoreFilterError(msg)

        joiner = " and " if operator == "AND" else " or "
        nested = (_normalize_filters(condition, content_field=content_field) for condition in conditions)
        return f"( {joiner.join(nested)} )"

    if operator == "NOT":
        conditions = filters.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 1:
            msg = "NOT filters must contain exactly one nested condition"
            raise VespaDocumentStoreFilterError(msg)
        return f"!( {_normalize_filters(conditions[0], content_field=content_field)} )"

    field = filters.get("field")
    comparison = filters.get("operator")
    if not isinstance(field, str) or not isinstance(comparison, str):
        msg = "Leaf filters must contain 'field' and 'operator'"
        raise VespaDocumentStoreFilterError(msg)

    return _convert_comparison(field, comparison, filters.get("value"), content_field=content_field)
