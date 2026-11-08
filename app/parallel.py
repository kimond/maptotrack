"""Parallel helpers.

* `prefetch(fn, ...)` starts a download (or any slow I/O) in the background and returns a Future.
* `io_map(fn, items)` runs I/O-bound work (HTTP downloads, file reads) on several threads.
* `pmap(fn, items)` runs CPU-bound work on the CPU pool. numpy, scipy.ndimage, Pillow, GDAL and GEOS
  release Python's GIL while they compute, so plain threads use several cores without copying the big
  arrays to other processes (which would cost more than it saves, and is slow to start on Windows).

Threads also keep things simple: one shared memory, no pickling, and Ctrl+C / job errors behave as before.
Set MAPTOTRACK_WORKERS=1 to run everything sequentially (debugging), or another number to cap the cores.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

_CPU_PREFIX = "mt-cpu"
_lock = threading.Lock()
_cpu_pool = None
_bg_pool = None


def workers() -> int:
    env = os.environ.get("MAPTOTRACK_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 2)


def _cpu():
    global _cpu_pool
    with _lock:
        if _cpu_pool is None:
            _cpu_pool = ThreadPoolExecutor(max_workers=workers(), thread_name_prefix=_CPU_PREFIX)
        return _cpu_pool


def _bg():
    global _bg_pool
    with _lock:
        if _bg_pool is None:
            _bg_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mt-bg")
        return _bg_pool


def _inside_cpu_pool() -> bool:
    return threading.current_thread().name.startswith(_CPU_PREFIX)


def pmap(fn, items) -> list:
    """Ordered map on the CPU pool. Runs inline when there is nothing to gain (1 item, 1 worker, or
    already inside a CPU-pool task - nesting would risk every worker waiting on the others)."""
    items = list(items)
    if len(items) <= 1 or workers() <= 1 or _inside_cpu_pool():
        return [fn(it) for it in items]
    return list(_cpu().map(fn, items))


def io_map(fn, items, threads: int = 6) -> list:
    """Ordered map for downloads / file I/O (own short-lived pool, so it can be used anywhere)."""
    items = list(items)
    if len(items) <= 1 or workers() <= 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(max_workers=min(threads, len(items)), thread_name_prefix="mt-io") as ex:
        return list(ex.map(fn, items))


def prefetch(fn, *args, **kwargs) -> Future:
    """Start fn(*args, **kwargs) in the background; .result() re-raises its exception, if any."""
    if workers() <= 1:
        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as exc:      # noqa: BLE001 - handed to the caller through the Future
            f.set_exception(exc)
        return f
    return _bg().submit(fn, *args, **kwargs)
