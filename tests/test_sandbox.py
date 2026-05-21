"""LocalBackend 单测（阶段 1 任务 1.7）。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from worker.sandbox import LocalBackend, SandboxHandle


@pytest.fixture
def backend(tmp_path: Path) -> LocalBackend:
    return LocalBackend(root_dir=tmp_path / "sb_root")


async def test_create_writes_context_package(backend: LocalBackend) -> None:
    h = await backend.create("hello context")
    assert h.backend == "local"
    assert h.sandbox_id
    workdir = Path(h.metadata["workdir"])
    assert workdir.exists()
    assert (workdir / "context.txt").read_text() == "hello context"
    await backend.destroy(h)


async def test_destroy_cleans_workdir(backend: LocalBackend) -> None:
    h = await backend.create("x")
    workdir = Path(h.metadata["workdir"])
    await backend.destroy(h)
    assert not workdir.exists()


async def test_exec_command_returns_stdout(backend: LocalBackend) -> None:
    h = await backend.create("")
    out = await backend.exec_command(h, "echo hello-sandbox")
    assert "hello-sandbox" in out
    await backend.destroy(h)


async def test_run_code_executes_python(backend: LocalBackend) -> None:
    h = await backend.create("")
    out = await backend.run_code(h, "print('from-inside', 2 + 3)")
    assert "from-inside 5" in out
    await backend.destroy(h)


async def test_read_write_file_roundtrip(backend: LocalBackend) -> None:
    h = await backend.create("")
    await backend.write_file(h, "sub/data.txt", "payload")
    got = await backend.read_file(h, "sub/data.txt")
    assert got == "payload"
    await backend.destroy(h)


async def test_path_escape_rejected(backend: LocalBackend) -> None:
    h = await backend.create("")
    with pytest.raises(ValueError):
        await backend.write_file(h, "../escape.txt", "no")
    await backend.destroy(h)


async def test_cancel_terminates_long_command(backend: LocalBackend) -> None:
    h = await backend.create("")

    async def long_run() -> str:
        return await backend.exec_command(
            h, f"{sys.executable} -c \"import time; time.sleep(10)\""
        )

    task = asyncio.create_task(long_run())
    await asyncio.sleep(0.3)
    ok = await backend.cancel(h, timeout=5.0)
    assert ok
    with pytest.raises(asyncio.CancelledError):
        await task
    await backend.destroy(h)


async def test_cancel_idempotent_when_no_proc(backend: LocalBackend) -> None:
    h = await backend.create("")
    assert await backend.cancel(h) is True
    await backend.destroy(h)
