import functools
import os
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import IO, Any, Generic, Optional, TextIO, TypeAlias, TypeVar, overload

import log
import numpy as np
from numpy.typing import NDArray
from typing_extensions import ParamSpec, Self, TypeAliasType

T = TypeVar("T")
P = ParamSpec("R")
R = TypeVar("R")
Sentinel = TypeVar("Sentinel")

_ParamSpecCallable: TypeAlias = Callable[P, R]
_TypeOrAlias: TypeAlias = type | TypeAliasType

FOLDER_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = FOLDER_DIR / "checkpoints"

_NO_ARG_PROVIDED = object()
_TIME_MEASUREMENTS = [("hour", 3600), ("minute", 60), ("second", 1)]


def flatten_dim(dim: int | tuple[int]) -> int:
    return dim[0] if isinstance(dim, tuple) else dim


def numpy_getitem(collection: T, indicies: NDArray[np.int32], *, index: bool) -> T:
    return collection[indicies] if index else collection


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
    iterable: Iterable[T], sentinel: Optional[Sentinel] = None
) -> Callable[[], Optional[T | Sentinel]]:
    iterator = iter(iterable)
    def iterator_func() -> Optional[T | Sentinel]:
        return next(iterator, sentinel)
    return iterator_func


def andjoin(iterable: Iterable, separator: str = ", ") -> str:
    *elements, last = iterable
    if not elements:
        return str(last)
    return f"{separator.join(elements)} and {last}"


def format_time(seconds: float, include_milliseconds: bool) -> str:
    parts = []
    for unit, unit_in_seconds in _TIME_MEASUREMENTS:
        unit_quantity, seconds = divmod(seconds, unit_in_seconds)
        if unit_quantity > 0:
            if unit_quantity != 1:
                unit += "s"
            parts.append(f"{int(unit_quantity)} {unit}")

    if parts and include_milliseconds:
        milliseconds = int(seconds * 1000)
        if milliseconds > 0:
            unit = "milisecond" if milliseconds == 1 else "miliseconds"
            parts.append(f"{milliseconds} {unit}")

    return andjoin(parts) if parts else "<1 second"


@overload
def timed(obj: _ParamSpecCallable, /) -> _ParamSpecCallable: ...


@overload
def timed(
    *,
    include_milliseconds: bool = False
) -> Callable[[_ParamSpecCallable], _ParamSpecCallable]: ...


def timed(obj: _ParamSpecCallable | object = _NO_ARG_PROVIDED, /, *, include_milliseconds: bool = False):
    def decorator(func: _ParamSpecCallable):
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = format_time(end_time - start_time, include_milliseconds)
            log(f"{func.__name__!r} took", elapsed_time, "to execute")
            return result
        return wrapper

    assert callable(obj) or obj is _NO_ARG_PROVIDED
    if obj is _NO_ARG_PROVIDED:
        return decorator
    return decorator(obj)


def dataclass_fieldnames(cls: type) -> set[str]:
    assert is_dataclass(cls), "This decorator only supports dataclasses"
    return set(field.name for field in fields(cls))


def filter_attributes_only(values: dict[str, Any], cls: type):
    assert is_dataclass(cls), "This decorator only supports dataclasses"
    fieldnames = dataclass_fieldnames(cls)
    return {
        key: value
        for key, value in values.items()
        if key in fieldnames
    }


def convert_to_cmd_args(obj: Any) -> list[str]:
    args = []
    for key, value in vars(obj).items():
        if value is not None:
            args.append(f"--{key}")
            args.append(repr(value))
    return args


def non_destructive_mutation(cls: type[T]) -> type[T]:
    assert is_dataclass(cls), "This decorator only supports dataclasses"

    def with_values(self: T, **replacements: Any) -> T:
        values = asdict(self)
        for name, value in replacements.items():
            if name not in values:
                raise ValueError(f"Unknown attribute {name!r}")
            values[name] = value
        return cls(**values)

    cls.with_values = with_values
    return cls


class NonDestructiveMutation:
    # Let's help the typechecker here with this 'mixin' to support the decorator
    def with_values(self, **values: Any) -> Self:
        raise NotImplementedError


class CountedInstanceCreation:
    def __init__(self):
        self.n_accesses = 0

    def __get__(self, *_) -> int:
        n_accesses = self.n_accesses
        self.n_accesses += 1
        return n_accesses


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
        # This should be the case anyway
        assert hasattr(generic_arg, "__instancecheck__")
        assert isinstance(generic_arg, type)
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


class TerminalLineClear:
    def __init__(
        self,
        *,
        reset_after_lines: int,
        jupyter_notebook: bool = True,
        start_after_seconds: int = 0
    ):
        self.reset_after_lines = reset_after_lines
        self.line_count = 0
        self.start_after_seconds = start_after_seconds
        self.original_stdout: Optional[TextIO] = None
        self.jupyter_notebook = jupyter_notebook
        self.threads: list[threading.Thread] = []
        self.start_thread: Optional[threading.Thread] = None
        self.start_event = threading.Event()
        self.lock = threading.Lock()

    def __enter__(self) -> Self:
        def start_thread():
            time.sleep(self.start_after_seconds)
            self.start_event.set()

        self.start_thread = threading.Thread(target=start_thread)
        self.start_thread.start()
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_: Any):
        sys.stdout = self.original_stdout
        for thread in self.threads:
            thread.join()

    def add_process(self, process: subprocess.Popen):
        def process_stream(stream: IO):
            for line in stream:
                if self.start_event.is_set():
                    with self.lock:
                        self.write(line)

        thread = threading.Thread(
            target=process_stream,
            args=(process.stdout,)
        )
        thread.start()
        self.threads.append(thread)

    def write(self, text: str):
        assert self.original_stdout is not None
        if self.line_count >= self.reset_after_lines:
            if self.jupyter_notebook:
                from IPython.display import clear_output
                clear_output(wait=True)
            else:
                os.system("cls" if os.name == "nt" else "clear")
            self.line_count = 0

        newlines = text.count('\n')
        self.line_count += newlines
        self.original_stdout.write(text)

    def flush(self):
        assert self.original_stdout is not None
        self.original_stdout.flush()
