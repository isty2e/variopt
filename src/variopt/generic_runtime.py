"""Runtime compatibility helpers for generic dataclasses."""

from dataclasses import dataclass, field, fields


@dataclass(frozen=True, slots=True, init=False)
class FrozenGenericSlotsCompat:
    """Provide a slot for ``typing`` runtime generic metadata.

    Notes
    -----
    Python 3.11 assigns ``__orig_class__`` when constructing subscripted
    generic aliases like ``Bank[int](...)``. Frozen slotted dataclasses raise
    at that assignment unless they declare the attribute explicitly.
    """

    __orig_class__: object | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __getstate__(self) -> list[object | None]:
        """Serialize slotted dataclasses without requiring ``__orig_class__``.

        Returns
        -------
        list[object | None]
            Field values in dataclass field order. Missing ``__orig_class__``
            is normalized to ``None`` so unsubscripted generic instances remain
            picklable under Python 3.11.
        """
        return [getattr(self, dataclass_field.name, None) for dataclass_field in fields(self)]

    def __setstate__(self, state: list[object | None]) -> None:
        """Restore one slotted dataclass state emitted by :meth:`__getstate__`.

        Parameters
        ----------
        state : list[object | None]
            Field values aligned with dataclass field order.
        """
        for dataclass_field, value in zip(fields(self), state):
            object.__setattr__(self, dataclass_field.name, value)
