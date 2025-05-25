import functools
import random
import time
import types
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from typing import Any, Generic, TypeAlias, TypeVar

import log
import numpy as np
import torch
from typing_extensions import Self, TypeAliasType

T = TypeVar("T")
Sentinel = TypeVar("Sentinel")

_Func = Callable[..., Any]
_TypeOrAlias: TypeAlias = type | TypeAliasType


def flatten_dim(dim: int | tuple[int]) -> int:
    return dim[0] if isinstance(dim, tuple) else dim


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_instance(obj: object, type_or_alias: _TypeOrAlias):
    def assertion(value: bool, expected: _TypeOrAlias, actual: _TypeOrAlias):
        assert value, log.colours.RED(
            f"Expected {expected.__name__!r}, got {actual.__name__!r}"
        )

    if isinstance(obj, np.ndarray):
        assertion(obj.dtype.type is type_or_alias, type_or_alias, obj.dtype.type)
        return

    unaliased = type_or_alias
    if isinstance(type_or_alias, TypeAliasType):
        unaliased = type_or_alias.__value__

    cannot_check = False
    reason = None

    if isinstance(unaliased, types.GenericAlias):
        cannot_check = True
        reason = "the provided typealias cannot be a GenericAlias"
    else:
        try:
            result = isinstance(obj, unaliased)
        except TypeError as err:
            cannot_check = True
            reason = err

    if cannot_check:
        log.warn("Cannot check if object is", unaliased, "because", reason)
    else:
        assertion(result, unaliased, type(obj))


def iter_factory(
    iterable: Iterable[T], sentinel: Sentinel | None = None
) -> Callable[[], T | Sentinel | None]:
    iterator = iter(iterable)

    def iterator_func() -> T | Sentinel | None:
        return next(iterator, sentinel)

    return iterator_func


def andjoin(iterable: Iterable, separator: str = ", ") -> str:
    *elements, last = iterable
    if not elements:
        return str(last)
    return f"{separator.join(elements)} and {last}"


def format_time(seconds: float) -> str:
    measurements = [("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []

    for measurement_name, measurement_in_seconds in measurements:
        measurement_quantity, seconds = divmod(seconds, measurement_in_seconds)
        if measurement_quantity > 0:
            if measurement_quantity != 1:
                measurement_name += "s"
            parts.append(f"{int(measurement_quantity)} {measurement_name}")

    return andjoin(parts) if parts else "0 seconds"


def timed(func: _Func):
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        log(f"{func.__name__!r} took", format_time(end_time - start_time), "to execute")
        return result

    return wrapper


class AttributeIterable(Generic[T]):
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        self = super().__new__(cls)
        try:
            generic_arg = self.__orig_bases__[0].__args__[0]
        except (AttributeError, IndexError, TypeError) as exc:
            raise TypeError(
                f"Unable to retrieve generic parameter of instance {self!r}"
            ) from exc

        # We have essentially 'reified' T by finding its specialisation
        # at runtime, that is, for AttributeIterable[list], finding
        # the actual 'list' type object
        assert hasattr(self, "__dict__")
        assert isinstance(generic_arg, type)
        # This should be the case anyway
        assert hasattr(generic_arg, "__instancecheck__")
        self._T = generic_arg
        return self

    def attribute_values(self) -> Iterator[T]:
        yield from filter(
            lambda value: isinstance(value, self._T), self.__dict__.values()
        )

    def attributes_by_name(self) -> Iterator[tuple[str, T]]:
        for name, value in self.__dict__.items():
            if isinstance(value, self._T):
                yield name, value


class AttributeUnpackable:
    __slots__ = ()

    def __iter__(self) -> Iterator:
        if hasattr(self, "__slots__"):
            yield from map(partial(getattr, self), self.__slots__)
        else:
            yield from self.__dict__.values()
