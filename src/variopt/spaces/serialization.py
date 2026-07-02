"""JSON-safe codecs for canonical structured-space candidates."""

from collections.abc import Mapping
from math import isfinite
from typing import TypeGuard

from ..json_types import JSONValue
from .composites.records import RecordCandidate
from .types import SpaceCandidateValue

_BYTES_MARKER = "__variopt_bytes__"
_BYTEARRAY_MARKER = "__variopt_bytearray__"
_MAPPING_MARKER = "__variopt_mapping__"
_MAPPING_FORMAT = "variopt.mapping"
_MAPPING_FORMAT_FIELD = "format"
_MAPPING_ITEMS_FIELD = "items"
_RESERVED_MARKER_KEYS = frozenset(
    {
        _BYTES_MARKER,
        _BYTEARRAY_MARKER,
        _MAPPING_MARKER,
    },
)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_reserved_marker_mapping(mapping: Mapping[object, object]) -> bool:
    return len(mapping) == 1 and next(iter(mapping)) in _RESERVED_MARKER_KEYS


def _encode_mapping(candidate: Mapping[object, object]) -> JSONValue:
    encoded_mapping: dict[str, JSONValue] = {}
    for key, value in candidate.items():
        if not isinstance(key, str):
            msg = "structured candidate mapping keys must be strings"
            raise TypeError(msg)
        encoded_mapping[key] = _space_candidate_to_json_value(value)

    if not _is_reserved_marker_mapping(candidate):
        return encoded_mapping

    encoded_items: list[JSONValue] = [
        [key, value]
        for key, value in encoded_mapping.items()
    ]
    return {
        _MAPPING_MARKER: {
            _MAPPING_FORMAT_FIELD: _MAPPING_FORMAT,
            _MAPPING_ITEMS_FIELD: encoded_items,
        },
    }


def _decode_mapping_items(
    items: list[object],
    *,
    record_candidates: bool,
) -> SpaceCandidateValue:
    decoded_mapping: dict[str, SpaceCandidateValue] = {}
    for item in items:
        if not _is_object_list(item) or len(item) != 2:
            msg = "mapping candidate payload items must be two-item arrays"
            raise TypeError(msg)
        key, value = item
        if not isinstance(key, str):
            msg = "mapping candidate payload keys must be strings"
            raise TypeError(msg)
        if key in decoded_mapping:
            msg = "mapping candidate payload keys must be unique"
            raise ValueError(msg)
        decoded_mapping[key] = _space_candidate_from_json_value(
            value,
            record_candidates=record_candidates,
        )
    if record_candidates:
        return RecordCandidate(entries=tuple(decoded_mapping.items()))
    return decoded_mapping


def _decode_mapping(
    data: Mapping[object, object],
    *,
    record_candidates: bool,
) -> SpaceCandidateValue:
    decoded_mapping: dict[str, SpaceCandidateValue] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            msg = "mapping candidate payload keys must be strings"
            raise TypeError(msg)
        decoded_mapping[key] = _space_candidate_from_json_value(
            value,
            record_candidates=record_candidates,
        )
    if record_candidates:
        return RecordCandidate(entries=tuple(decoded_mapping.items()))
    return decoded_mapping


def _decode_escaped_mapping(
    payload: object,
    *,
    record_candidates: bool,
) -> SpaceCandidateValue | None:
    if not _is_object_mapping(payload):
        return None
    if payload.get(_MAPPING_FORMAT_FIELD) != _MAPPING_FORMAT:
        return None

    raw_items = payload.get(_MAPPING_ITEMS_FIELD)
    if not _is_object_list(raw_items):
        msg = "mapping candidate payload requires an items array"
        raise TypeError(msg)
    return _decode_mapping_items(raw_items, record_candidates=record_candidates)


def space_candidate_to_dict(candidate: SpaceCandidateValue) -> JSONValue:
    """Return a JSON-safe representation of one canonical structured candidate.

    Parameters
    ----------
    candidate : SpaceCandidateValue
        Canonical structured candidate to encode.

    Returns
    -------
    JSONValue
        JSON-safe candidate payload.

    Raises
    ------
    TypeError
        If ``candidate`` is not a supported canonical structured candidate.
    ValueError
        If ``candidate`` contains non-finite floating-point values.
    """
    return _space_candidate_to_json_value(candidate)


def _space_candidate_to_json_value(candidate: object) -> JSONValue:
    if isinstance(candidate, bytearray):
        return {_BYTEARRAY_MARKER: candidate.hex()}

    if isinstance(candidate, bytes):
        return {_BYTES_MARKER: candidate.hex()}

    if isinstance(candidate, float):
        if not isfinite(candidate):
            msg = "float structured candidate values must be finite"
            raise ValueError(msg)
        return candidate

    if isinstance(candidate, (bool, int, str)):
        return candidate

    if _is_object_tuple(candidate):
        return [_space_candidate_to_json_value(value) for value in candidate]

    if _is_object_mapping(candidate):
        return _encode_mapping(candidate)

    msg = "expected a supported structured candidate value"
    raise TypeError(msg)


def space_candidate_from_dict(
    data: JSONValue,
    *,
    record_candidates: bool = False,
) -> SpaceCandidateValue:
    """Return one structured candidate payload from a JSON-safe payload.

    Parameters
    ----------
    data : JSONValue
        JSON-safe structured candidate payload.
    record_candidates : bool, default=False
        When ``True``, JSON object payloads are rebuilt as
        :class:`~variopt.spaces.RecordCandidate` values instead of plain
        mappings. Use this for built-in structured spaces that contain
        ``RecordSpace`` nodes.

    Returns
    -------
    SpaceCandidateValue
        Decoded structured candidate payload.

    Raises
    ------
    TypeError
        If ``data`` is not a supported structured candidate payload.
    ValueError
        If a bytes payload carries invalid hexadecimal content or a scalar
        float is non-finite.
    """
    return _space_candidate_from_json_value(data, record_candidates=record_candidates)


def _space_candidate_from_json_value(
    data: object,
    *,
    record_candidates: bool,
) -> SpaceCandidateValue:
    if _is_object_list(data):
        return tuple(
            _space_candidate_from_json_value(value, record_candidates=record_candidates)
            for value in data
        )

    if _is_object_mapping(data):
        if set(data.keys()) == {_BYTES_MARKER}:
            raw_hex = data[_BYTES_MARKER]
            if not isinstance(raw_hex, str):
                msg = "bytes candidate payload must carry a hexadecimal string"
                raise TypeError(msg)
            return bytes.fromhex(raw_hex)

        if set(data.keys()) == {_BYTEARRAY_MARKER}:
            raw_hex = data[_BYTEARRAY_MARKER]
            if not isinstance(raw_hex, str):
                msg = "bytearray candidate payload must carry a hexadecimal string"
                raise TypeError(msg)
            return bytearray.fromhex(raw_hex)

        if set(data.keys()) == {_MAPPING_MARKER}:
            decoded_mapping = _decode_escaped_mapping(
                data[_MAPPING_MARKER],
                record_candidates=record_candidates,
            )
            if decoded_mapping is not None:
                return decoded_mapping

        return _decode_mapping(data, record_candidates=record_candidates)

    if data is None:
        msg = "structured candidate payload must not be null"
        raise TypeError(msg)

    if isinstance(data, float):
        if not isfinite(data):
            msg = "float structured candidate payload values must be finite"
            raise ValueError(msg)
        return data

    if isinstance(data, (bool, int, str)):
        return data

    msg = "expected a supported structured candidate payload"
    raise TypeError(msg)
