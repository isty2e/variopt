"""Immutable canonical JSON values for CSA configuration manifests."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import TypeAlias

from .....json_types import JSONDict, JSONScalar, JSONValue

MAX_CANONICAL_JSON_DEPTH = 256


@dataclass(frozen=True, slots=True, init=False)
class FrozenJSONScalar:
    """Immutable JSON scalar with type-exact equality semantics."""

    value: JSONScalar = field(compare=False, hash=False)
    canonical_token: str = field(repr=False)

    def __init__(self, value: JSONScalar) -> None:
        """Store one finite, exact built-in JSON scalar.

        Parameters
        ----------
        value : JSONScalar
            Scalar value to freeze.

        Raises
        ------
        TypeError
            If ``value`` is not an exact built-in JSON scalar.
        ValueError
            If ``value`` is a non-finite float.
        """
        if value is not None and type(value) not in {bool, int, float, str}:
            msg = "manifest JSON scalars must use exact built-in JSON types"
            raise TypeError(msg)
        if isinstance(value, float) and not isfinite(value):
            msg = "manifest JSON floats must be finite"
            raise ValueError(msg)

        canonical_token = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "canonical_token", canonical_token)


@dataclass(frozen=True, slots=True)
class FrozenJSONArray:
    """Immutable ordered JSON array."""

    values: tuple["FrozenJSONValue", ...]


@dataclass(frozen=True, slots=True)
class FrozenJSONObject:
    """Immutable JSON object with keys stored in canonical order."""

    entries: tuple[tuple[str, "FrozenJSONValue"], ...]


FrozenJSONValue: TypeAlias = FrozenJSONScalar | FrozenJSONArray | FrozenJSONObject


def freeze_json_object(
    data: Mapping[str, JSONValue],
    *,
    field_name: str,
) -> FrozenJSONObject:
    """Return an immutable canonical snapshot of a JSON object."""
    return _freeze_json_mapping(
        data,
        active_container_ids=set(),
        depth=0,
        field_name=field_name,
    )


def thaw_json_object(value: FrozenJSONObject) -> JSONDict:
    """Return a fresh mutable JSON object from an immutable snapshot."""
    return {key: thaw_json_value(child_value) for key, child_value in value.entries}


def thaw_json_value(value: FrozenJSONValue) -> JSONValue:
    """Return a fresh mutable JSON value from an immutable snapshot."""
    if isinstance(value, FrozenJSONScalar):
        return value.value
    if isinstance(value, FrozenJSONArray):
        return [thaw_json_value(child_value) for child_value in value.values]
    return thaw_json_object(value)


def _freeze_json_value(
    value: JSONValue,
    *,
    active_container_ids: set[int],
    depth: int,
    field_name: str,
) -> FrozenJSONValue:
    if depth > MAX_CANONICAL_JSON_DEPTH:
        msg = f"{field_name} exceeds the supported manifest JSON depth"
        raise ValueError(msg)

    if value is None:
        return FrozenJSONScalar(value)

    if isinstance(value, bool):
        if type(value) is not bool:
            msg = f"{field_name} must use an exact built-in JSON type"
            raise TypeError(msg)
        return FrozenJSONScalar(value)

    if isinstance(value, int):
        if type(value) is not int:
            msg = f"{field_name} must use an exact built-in JSON type"
            raise TypeError(msg)
        return FrozenJSONScalar(value)

    if isinstance(value, float):
        if type(value) is not float:
            msg = f"{field_name} must use an exact built-in JSON type"
            raise TypeError(msg)
        if not isfinite(value):
            msg = f"{field_name} must be finite"
            raise ValueError(msg)
        return FrozenJSONScalar(value)

    if isinstance(value, str):
        if type(value) is not str:
            msg = f"{field_name} must use an exact built-in JSON type"
            raise TypeError(msg)
        validate_utf8_string(value, field_name=field_name)
        return FrozenJSONScalar(value)

    if isinstance(value, list):
        if type(value) is not list:
            msg = f"{field_name} must use an exact built-in JSON container"
            raise TypeError(msg)
        container_id = _enter_container(
            value,
            active_container_ids=active_container_ids,
            field_name=field_name,
        )
        try:
            return FrozenJSONArray(
                values=tuple(
                    _freeze_json_value(
                        child_value,
                        active_container_ids=active_container_ids,
                        depth=depth + 1,
                        field_name=f"{field_name}[{index}]",
                    )
                    for index, child_value in enumerate(value)
                ),
            )
        finally:
            active_container_ids.remove(container_id)

    if isinstance(value, dict):
        if type(value) is not dict:
            msg = f"{field_name} must use an exact built-in JSON container"
            raise TypeError(msg)
        return _freeze_json_mapping(
            value,
            active_container_ids=active_container_ids,
            depth=depth,
            field_name=field_name,
        )

    msg = f"{field_name} must be a JSON value"
    raise TypeError(msg)


def _freeze_json_mapping(
    data: Mapping[str, JSONValue],
    *,
    active_container_ids: set[int],
    depth: int,
    field_name: str,
) -> FrozenJSONObject:
    if depth > MAX_CANONICAL_JSON_DEPTH:
        msg = f"{field_name} exceeds the supported manifest JSON depth"
        raise ValueError(msg)

    container_id = _enter_container(
        data,
        active_container_ids=active_container_ids,
        field_name=field_name,
    )
    try:
        entries: list[tuple[str, FrozenJSONValue]] = []
        for key, child_value in data.items():
            if type(key) is not str:
                msg = f"{field_name} keys must be exact built-in strings"
                raise TypeError(msg)
            validate_utf8_string(key, field_name=f"{field_name} key")
            encoded_key = json.dumps(key, ensure_ascii=False)
            entries.append(
                (
                    key,
                    _freeze_json_value(
                        child_value,
                        active_container_ids=active_container_ids,
                        depth=depth + 1,
                        field_name=f"{field_name}[{encoded_key}]",
                    ),
                ),
            )
        entries.sort(key=lambda entry: entry[0])
        return FrozenJSONObject(entries=tuple(entries))
    finally:
        active_container_ids.remove(container_id)


def _enter_container(
    container: Mapping[str, JSONValue] | list[JSONValue],
    *,
    active_container_ids: set[int],
    field_name: str,
) -> int:
    container_id = id(container)
    if container_id in active_container_ids:
        msg = f"{field_name} must not contain a container cycle"
        raise ValueError(msg)
    active_container_ids.add(container_id)
    return container_id


def validate_utf8_string(value: str, *, field_name: str) -> None:
    """Reject strings that cannot be represented as UTF-8."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        msg = f"{field_name} must be valid UTF-8 text"
        raise ValueError(msg) from error
