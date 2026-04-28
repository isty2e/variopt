"""JSON-safe codecs for canonical structured-space candidates."""

from collections.abc import Mapping

from ..json_types import JSONValue
from .types import SpaceCandidateValue

_BYTES_MARKER = "__variopt_bytes__"
_BYTEARRAY_MARKER = "__variopt_bytearray__"


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
    """
    if isinstance(candidate, bytearray):
        return {_BYTEARRAY_MARKER: candidate.hex()}

    if isinstance(candidate, bytes):
        return {_BYTES_MARKER: candidate.hex()}

    if isinstance(candidate, tuple):
        return [space_candidate_to_dict(value) for value in candidate]

    if isinstance(candidate, Mapping):
        encoded_mapping: dict[str, JSONValue] = {}
        for key, value in candidate.items():
            encoded_mapping[key] = space_candidate_to_dict(value)
        return encoded_mapping

    return candidate


def space_candidate_from_dict(data: JSONValue) -> SpaceCandidateValue:
    """Return one canonical structured candidate from a JSON-safe payload.

    Parameters
    ----------
    data : JSONValue
        JSON-safe structured candidate payload.

    Returns
    -------
    SpaceCandidateValue
        Decoded canonical structured candidate.

    Raises
    ------
    TypeError
        If ``data`` is not a supported structured candidate payload.
    ValueError
        If a bytes payload carries invalid hexadecimal content.
    """
    if isinstance(data, list):
        return tuple(space_candidate_from_dict(value) for value in data)

    if isinstance(data, dict):
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

        decoded_mapping: dict[str, SpaceCandidateValue] = {}
        for key, value in data.items():
            decoded_mapping[key] = space_candidate_from_dict(value)
        return decoded_mapping

    if data is None:
        msg = "structured candidate payload must not be null"
        raise TypeError(msg)

    return data
