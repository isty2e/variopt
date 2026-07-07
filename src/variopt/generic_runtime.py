"""Runtime compatibility helpers for generic dataclasses."""

from collections.abc import Callable
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


def frozen_generic_slots_compat_getstate(
    self: FrozenGenericSlotsCompat,
) -> list[object | None]:
    """Serialize slotted dataclasses without requiring ``__orig_class__``.

    Returns
    -------
    list[object | None]
        Field values in dataclass field order. Missing ``__orig_class__``
        is normalized to ``None`` so unsubscripted generic instances remain
        picklable under Python 3.11.
    """
    return [
        getattr(self, dataclass_field.name, None) for dataclass_field in fields(self)
    ]


def frozen_generic_slots_compat_setstate(
    self: FrozenGenericSlotsCompat,
    state: list[object | None],
) -> None:
    """Restore one slotted dataclass state emitted by :meth:`__getstate__`.

    Parameters
    ----------
    state : list[object | None]
        Field values aligned with dataclass field order.
    """
    for dataclass_field, value in zip(fields(self), state):
        object.__setattr__(self, dataclass_field.name, value)


def create_frozen_generic_slots_pickle_installer(
    getstate: Callable[[FrozenGenericSlotsCompat], list[object | None]],
    setstate: Callable[[FrozenGenericSlotsCompat, list[object | None]], None],
) -> Callable[[type[FrozenGenericSlotsCompat]], None]:
    """Return the standard pickle-hook installer while keeping hooks private."""

    def install_frozen_generic_slots_pickle(
        cls: type[FrozenGenericSlotsCompat],
    ) -> None:
        """Install tolerant pickle hooks on one frozen slotted dataclass.

        Parameters
        ----------
        cls : type[FrozenGenericSlotsCompat]
            Frozen slotted dataclass type that should tolerate a missing
            ``__orig_class__`` slot during pickling.

        Notes
        -----
        ``@dataclass(frozen=True, slots=True)`` installs pickle hooks on every
        dataclass subclass. Apply this helper after the subclass decorator has
        run when a custom ``__init__`` cannot rely on dataclasses to populate
        inherited runtime generic metadata.
        """
        setattr(cls, "__getstate__", getstate)
        setattr(cls, "__setstate__", setstate)

    return install_frozen_generic_slots_pickle


install_frozen_generic_slots_pickle = create_frozen_generic_slots_pickle_installer(
    frozen_generic_slots_compat_getstate,
    frozen_generic_slots_compat_setstate,
)
install_frozen_generic_slots_pickle(FrozenGenericSlotsCompat)

del create_frozen_generic_slots_pickle_installer
del frozen_generic_slots_compat_getstate
del frozen_generic_slots_compat_setstate
