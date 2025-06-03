import asyncio
import atexit
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, TypeAlias

import numpy as np
import torch
from colorama import Fore
from getkey import getkey
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Never

from src.dtypes import LayoutName
from src.misc import FOLDER_DIR

RUNS_DIR = FOLDER_DIR / "runs"
DELETE_PREVIOUS_TENSORBOARD_RUNS = True

_DEFAULT_TENSORBOARD_PORT = 6006

_TQDM_COLOURS = ["blue", "cyan", "green", "magenta", "red", "yellow"]

_chosen_colours: set[str] = set()
_writers: dict[str, SummaryWriter] = {}
_tensorboard_init = [False]

_device: Optional[torch.device] = [None]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def best_available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_device() -> torch.device:
    assert _device[0] is not None
    return _device[0]


def set_device(device: Optional[torch.device | str] = None):
    assert _device[0] is None
    if isinstance(device, str):
        device = torch.device(device)
    _device[0] = device or best_available_device()


def tensor_mean(values: list[Any] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.mean()
    else:
        return torch.tensor(values).mean()


def random_tqdm_colour() -> str:
    rand = random.Random()
    colour = rand.choice(_TQDM_COLOURS)
    while colour in _chosen_colours:
        colour = rand.choice(_TQDM_COLOURS)
    _chosen_colours.add(colour)
    return colour


def get_proper_layout_name(layout_name: str | LayoutName) -> str:
    return ' '.join(layout_name.split('_')).title()


class NullWriter:
    def __getattr__(self, _: Any):
        def noop(*args: Any, **kwargs: Any):
            pass
        return noop


class GlobalState:
    current_episode: ClassVar[int] = 0

    def __init__(self) -> Never:
        raise Exception(f"{self.__class__.__name__!r} is not instantiable")


Writer: TypeAlias = SummaryWriter | NullWriter
WriterFactory: TypeAlias = Callable[[], Writer]

def create_writer(
    *,
    name: Optional[str] = None,
    predicate: Optional[Callable[..., bool]] = None,
    write: bool = True
) -> Writer:
    def writer_key() -> Optional[str]:
        return name

    if not write or (predicate is not None and not predicate()):
        return NullWriter()

    if name is not None:
        log_dir = RUNS_DIR / name
    else:
        log_dir = RUNS_DIR

    key = writer_key()
    if (writer := _writers.get(key)) is not None:
        return writer

    RUNS_DIR.mkdir(exist_ok=True)
    if DELETE_PREVIOUS_TENSORBOARD_RUNS and not _tensorboard_init[0]:
        _delete_previous_tensorboard_runs()
        _tensorboard_init[0] = True

    _writers[key] = writer = SummaryWriter(log_dir=log_dir)
    return writer


def close_writers():
    for writer in _writers.values():
        writer.close()


def _delete_previous_tensorboard_runs(runs_dir: Path = RUNS_DIR):
    def is_tensorboard_file(filename: str) -> bool:
        return filename.startswith("events")

    for dirpath, _, filenames in os.walk(runs_dir):
        for filename in filter(is_tensorboard_file, filenames):
            path = Path(dirpath).joinpath(filename)
            path.unlink()

    for path in filter(Path.is_dir, runs_dir.iterdir()):
        path.rmdir()


async def _create_tensorboard_process(
    log_dir: os.PathLike,
    port: int
) -> asyncio.subprocess.Process:
    process = await asyncio.create_subprocess_exec(
        "tensorboard",
        "--logdir",
        str(log_dir),
        "--host",
        "localhost",
        "--port",
        str(port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return process

start_event = asyncio.Event()

def _stop_tensorboard_on_keypress(process: asyncio.subprocess.Process):
    print(Fore.MAGENTA + "Press any key to stop TensorBoard..." + Fore.RESET)
    start_event.set()
    _ = getkey()
    process.kill()


async def _print_process_output(
    process: asyncio.subprocess.Process, descriptor: Literal["stdout", "stderr"]
):
    await start_event.wait()
    reader = process.stdout if descriptor == "stdout" else process.stderr
    assert reader is not None, f"Reader for {descriptor} is None"

    async for line in reader:
        colour = Fore.RED if descriptor == "stderr" else Fore.CYAN
        output = colour + line.decode().strip() + Fore.RESET
        print(output)


async def _run_tensorboard(log_dir: os.PathLike, port: int):
    try:
        process = await _create_tensorboard_process(log_dir, port)
        stdout_process = asyncio.create_task(
            _print_process_output(process, "stdout")
        )
        stderr_process = asyncio.create_task(
            _print_process_output(process, "stderr")
        )
        process_stop = asyncio.to_thread(
            _stop_tensorboard_on_keypress, process
        )
    finally:
        await asyncio.gather(
            stdout_process,
            stderr_process,
            process_stop,
            process.wait()
        )


def run_tensorboard(log_dir: os.PathLike = RUNS_DIR, port: Optional[int] = None):
    try:
        asyncio.run(_run_tensorboard(log_dir, port or _DEFAULT_TENSORBOARD_PORT))
    except KeyboardInterrupt:
        pass


def on_program_exit():
    close_writers()


atexit.register(on_program_exit)
