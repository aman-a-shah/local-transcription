"""Concurrency tests for the engine's self-healing task lane (`_Lane`).

These cover the exact failure that froze the app: a single-worker lane whose task
wedges in a never-returning call. The lane must (1) run tasks serially in order,
(2) rebuild itself so a wedged task can't jam the lane forever, and (3) shut down
in bounded time even while a task is stuck — so the quit path can never hang.

No microphone, key tap, or model needed; tasks are plain Python callables.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dictate.core import _Lane  # noqa: E402


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_runs_tasks_in_order():
    lane = _Lane("test-order", stall_seconds=999)
    out: list[int] = []
    for i in range(50):
        lane.submit(out.append, i)
    assert _wait_until(lambda: len(out) == 50), out
    assert out == list(range(50))  # strict FIFO, one at a time
    lane.shutdown()


def test_recovers_from_a_wedged_task():
    """A task stuck forever must not stop later tasks once the lane recovers."""
    lane = _Lane("test-wedge", stall_seconds=0.2)
    release = threading.Event()
    ran_after: list[str] = []

    lane.submit(release.wait)          # wedges the worker until we let go
    lane.submit(ran_after.append, "after")  # queued behind the wedged task

    # While wedged, the follow-up task cannot run.
    time.sleep(0.4)
    assert ran_after == [], "queued task ran despite a wedged worker"

    # The watchdog tick abandons the wedged worker and rebuilds the lane.
    assert lane.recover_if_stalled() is True
    # A fresh submission now runs on the new worker.
    lane.submit(ran_after.append, "recovered")
    assert _wait_until(lambda: "recovered" in ran_after), ran_after

    release.set()  # let the abandoned (daemon) worker unwind; it leaks harmlessly
    lane.shutdown()


def test_on_recover_hook_fires():
    fired: list[bool] = []
    lane = _Lane("test-hook", stall_seconds=0.1, on_recover=lambda: fired.append(True))
    block = threading.Event()
    lane.submit(block.wait)
    time.sleep(0.2)
    assert lane.recover_if_stalled() is True
    assert fired == [True]
    block.set()
    lane.shutdown()


def test_no_false_recovery_for_healthy_tasks():
    """A lane doing normal, fast work is never mistaken for wedged."""
    lane = _Lane("test-healthy", stall_seconds=0.5)
    for _ in range(20):
        lane.submit(time.sleep, 0.01)
    # Each task finishes well under the deadline, so recovery never triggers.
    for _ in range(10):
        assert lane.recover_if_stalled() is False
        time.sleep(0.02)
    lane.shutdown()


def test_shutdown_is_bounded_even_when_wedged():
    """The original quit-freeze: shutdown must return promptly despite a stuck task."""
    lane = _Lane("test-shutdown", stall_seconds=999)
    block = threading.Event()
    lane.submit(block.wait)  # wedge the worker
    time.sleep(0.1)

    t0 = time.monotonic()
    lane.shutdown(wait=True, timeout=0.5)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"shutdown blocked for {elapsed:.2f}s on a wedged worker"

    block.set()


def test_submit_after_shutdown_is_a_noop():
    lane = _Lane("test-closed", stall_seconds=999)
    lane.shutdown()
    ran: list[int] = []
    lane.submit(ran.append, 1)  # must not raise, must not run
    time.sleep(0.1)
    assert ran == []


if __name__ == "__main__":
    test_runs_tasks_in_order()
    print("✓ order")
    test_recovers_from_a_wedged_task()
    print("✓ recovery")
    test_on_recover_hook_fires()
    print("✓ on_recover hook")
    test_no_false_recovery_for_healthy_tasks()
    print("✓ no false recovery")
    test_shutdown_is_bounded_even_when_wedged()
    print("✓ bounded shutdown")
    test_submit_after_shutdown_is_a_noop()
    print("✓ submit after shutdown")
    print("\nAll lane checks passed.")
