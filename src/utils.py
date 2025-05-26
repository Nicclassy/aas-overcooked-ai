import asyncio
import os
import random
from typing import Any, Literal

import log
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.misc import FOLDER_DIR

_writer: SummaryWriter | None = None

LOG_DIR = FOLDER_DIR / "runs"

DEFAULT_TENSORBOARD_PORT = 6006
DELETE_PREVIOUS_TENSORBOARD_RUNS = True


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NullWriter:
    def __getattr__(self, _: Any):
        def noop(*args: Any, **kwargs: Any):
            pass

        return noop


def get_summary_writer(write: bool = True) -> SummaryWriter | NullWriter:
    global _writer
    if not write:
        return NullWriter()

    log_dir = LOG_DIR
    if _writer is not None:
        return _writer

    if not log_dir.exists():
        log_dir.mkdir()

    if DELETE_PREVIOUS_TENSORBOARD_RUNS:
        for path in filter(lambda p: p.name.startswith("events"), log_dir.iterdir()):
            path.unlink()

    _writer = SummaryWriter(log_dir=log_dir)
    return _writer


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


def run_tensorboard(log_dir: os.PathLike = LOG_DIR, port: int | None = None):
    try:
        asyncio.run(_run_tensorboard(log_dir, port or DEFAULT_TENSORBOARD_PORT))
    except KeyboardInterrupt:
        pass
