"""Shared JSON-safe type aliases and validators for snapshot codecs."""

from math import isfinite
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict: TypeAlias = dict[str, JSONValue]


def require_json_mapping(value: JSONValue, *, field_name: str) -> JSONDict:
    """Return one JSON object or raise.

    Parameters
    ----------
    value : JSONValue
        Raw JSON-safe value to validate.
    field_name : str
        Snapshot field name used in error messages.

    Returns
    -------
    JSONDict
        JSON object value.

    Raises
    ------
    TypeError
        If ``value`` is not a JSON object.
    """
    if not isinstance(value, dict):
        msg = f"{field_name} must be a JSON object"
        raise TypeError(msg)
    return value


def require_json_list(value: JSONValue, *, field_name: str) -> list[JSONValue]:
    """Return one JSON array or raise.

    Parameters
    ----------
    value : JSONValue
        Raw JSON-safe value to validate.
    field_name : str
        Snapshot field name used in error messages.

    Returns
    -------
    list[JSONValue]
        JSON array value.

    Raises
    ------
    TypeError
        If ``value`` is not a JSON array.
    """
    if not isinstance(value, list):
        msg = f"{field_name} must be a JSON array"
        raise TypeError(msg)
    return value


def require_json_str(value: JSONValue, *, field_name: str) -> str:
    """Return one JSON string or raise."""
    if not isinstance(value, str):
        msg = f"{field_name} must be a JSON string"
        raise TypeError(msg)
    return value


def require_json_optional_str(value: JSONValue, *, field_name: str) -> str | None:
    """Return one optional JSON string or raise."""
    if value is None:
        return None
    return require_json_str(value, field_name=field_name)


def require_json_int(value: JSONValue, *, field_name: str) -> int:
    """Return one JSON integer or raise."""
    if type(value) is not int:
        msg = f"{field_name} must be a JSON integer"
        raise TypeError(msg)
    return value


def require_json_int_or_str(value: JSONValue, *, field_name: str) -> int | str:
    """Return one JSON integer-or-string value or raise."""
    if type(value) is int:
        return value
    if isinstance(value, str):
        return value
    msg = f"{field_name} must be a JSON integer or string"
    raise TypeError(msg)


def require_json_bool(value: JSONValue, *, field_name: str) -> bool:
    """Return one JSON boolean or raise."""
    if not isinstance(value, bool):
        msg = f"{field_name} must be a JSON boolean"
        raise TypeError(msg)
    return value


def require_json_float(value: JSONValue, *, field_name: str) -> float:
    """Return one JSON numeric value as ``float`` or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field_name} must be a JSON number"
        raise TypeError(msg)
    return float(value)


def require_json_finite_float(value: JSONValue, *, field_name: str) -> float:
    """Return one finite JSON numeric value as ``float`` or raise."""
    number = require_json_float(value, field_name=field_name)
    if not isfinite(number):
        msg = f"{field_name} must be finite"
        raise ValueError(msg)
    return number


def require_json_optional_float(
    value: JSONValue,
    *,
    field_name: str,
) -> float | None:
    """Return one optional JSON numeric value or raise."""
    if value is None:
        return None
    return require_json_float(value, field_name=field_name)


def require_json_optional_finite_float(
    value: JSONValue,
    *,
    field_name: str,
) -> float | None:
    """Return one optional finite JSON numeric value or raise."""
    if value is None:
        return None
    return require_json_finite_float(value, field_name=field_name)
