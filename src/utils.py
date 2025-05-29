import asyncio
import atexit
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import log
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Never

from src.misc import FOLDER_DIR

RUNS_DIR = FOLDER_DIR / "runs"

_DEFAULT_TENSORBOARD_PORT = 6006
_DELETE_PREVIOUS_TENSORBOARD_RUNS = True

_writers: dict[str, SummaryWriter] = {}
_tensorboard_init = [False]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NullWriter:
    def __getattr__(self, _: Any):
        def noop(*args: Any, **kwargs: Any):
            pass

        return noop


class GlobalState:
    is_training: ClassVar[bool] = True
    game_number: ClassVar[int] = 0
    n_agents_created: ClassVar[int] = 0

    def __init__(self) -> Never:
        raise Exception(f"{self.__class__.__name__!r} is not instantiable")


def summary_writer_factory(
    *,
    agent_number: Optional[int] = None,
    game_number: Optional[int] = None,
    custom_name: Optional[str] = None,
    predicate: Optional[Callable[..., bool]] = None,
    write: bool = True,
) -> SummaryWriter | NullWriter:
    def writers_key() -> Optional[str]:
        if custom_name is not None:
            return custom_name
        if agent_number is not None and game_number is not None:
            return f"{agent_number}:{game_number}"
        if agent_number is not None:
            return str(agent_number)
        return None

    if GlobalState.is_training or not write or (predicate is not None and not predicate()):
        return NullWriter()

    if custom_name is not None:
        assert agent_number is None and game_number is None
    if game_number is not None:
        assert agent_number is not None

    if custom_name is not None:
        log_dir = RUNS_DIR / custom_name
    elif agent_number is not None and game_number is not None:
        log_dir = RUNS_DIR / f"Agent {agent_number} (Game {game_number})"
    elif agent_number is not None:
        log_dir = RUNS_DIR / f"Agent {agent_number}"
    else:
        log_dir = RUNS_DIR

    key = writers_key()
    if (writer := _writers.get(key)) is not None:
        return writer

    RUNS_DIR.mkdir(exist_ok=True)
    if _DELETE_PREVIOUS_TENSORBOARD_RUNS and not _tensorboard_init[0]:
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
    port: int,
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


async def _print_process_output(
    process: asyncio.subprocess.Process, descriptor: Literal["stdout", "stderr"]
):
    reader = process.stdout if descriptor == "stdout" else process.stderr
    assert reader is not None, f"Reader for {descriptor} is None"

    async for line in reader:
        colour = log.colours.RED if descriptor == "stderr" else log.colours.CYAN
        line = line.decode().strip()
        log.log(colour(line))


async def _run_tensorboard(log_dir: os.PathLike, port: int):
    process = await _create_tensorboard_process(log_dir, port)
    stdout_process = asyncio.create_task(_print_process_output(process, "stdout"))
    stderr_process = asyncio.create_task(_print_process_output(process, "stderr"))
    await asyncio.gather(stdout_process, stderr_process)
    await process.wait()


def run_tensorboard(log_dir: os.PathLike = RUNS_DIR, port: int | None = None):
    try:
        asyncio.run(_run_tensorboard(log_dir, port or _DEFAULT_TENSORBOARD_PORT))
    except KeyboardInterrupt:
        pass


def on_program_exit():
    close_writers()


atexit.register(on_program_exit)
