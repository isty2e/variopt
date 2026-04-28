"""Composite candidate value objects."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from typing_extensions import override

from ..types import SpaceCandidateValue


@dataclass(frozen=True)
class RecordCandidate(Mapping[str, SpaceCandidateValue]):
    """Immutable value object for record-shaped candidates.

    Parameters
    ----------
    entries : tuple[tuple[str, SpaceCandidateValue], ...]
        Canonical ordered field-value pairs.
    """

    entries: tuple[tuple[str, SpaceCandidateValue], ...]

    @override
    def __getitem__(self, key: str) -> SpaceCandidateValue:
        """Return a field value by name.

        Parameters
        ----------
        key : str
            Field name to look up.

        Returns
        -------
        SpaceCandidateValue
            Canonical value stored for ``key``.
        """
        for name, value in self.entries:
            if name == key:
                return value

        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[str]:
        """Iterate field names in canonical order.

        Returns
        -------
        Iterator[str]
            Iterator over field names in stored order.
        """
        for name, _ in self.entries:
            yield name

    @override
    def __len__(self) -> int:
        """Return the number of fields.

        Returns
        -------
        int
            Number of stored fields.
        """
        return len(self.entries)

    def as_dict(self) -> dict[str, SpaceCandidateValue]:
        """Return the candidate values as a plain dictionary copy.

        Returns
        -------
        dict[str, SpaceCandidateValue]
            Shallow dictionary copy of the stored entries.
        """
        return dict(self.entries)
