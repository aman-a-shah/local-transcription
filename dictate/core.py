"""The dictation engine, decoupled from any UI.

Holds the record -> transcribe -> inject pipeline and emits state changes through
a callback. Both the terminal front-end (`app.py`) and the menu-bar app
(`menu_app.py`) drive this same engine; they differ only in how they render the
states it emits.

States emitted via ``on_state(state, info)``:
    "ready"        model warmed, idle and waiting
    "listening"    fn held, mic capturing
    "transcribing" fn released, model running        info={"duration": s}
    "result"       text inserted                     info={"text", "elapsed", "duration"}
    "empty"        nothing recognized / tap too short
    "error"        transcription raised              info={"error": str}
    "idle"         returned to rest (follows result/empty/error)
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from .audio import Recorder
from .backends import create_transcriber
from .backends.base import has_speech
from .config import CONFIG
from .platforms import Feedback, inject
from .polish import polish
from .paths import log_path

StateCallback = Callable[[str, Optional[dict]], None]

_LOG_PATH = str(log_path())

# Per-lane stall deadlines (seconds). A task in a lane that outlives its deadline
# can only be *wedged* in a native call that never returned — no healthy task in
# that lane ever runs this long — so the watchdog rebuilds the lane around it.
# Generous enough that a legitimately slow operation never trips them:
#   capture — start/stop the mic stream; real tasks finish in well under a second.
#   inject  — clipboard write + ⌘V; sub-second in practice.
#   stt     — model inference on up to max_record_seconds of audio; kept large so
#             a slow machine on a long clip is never mistaken for a hang.
_CAPTURE_STALL = 15.0
_INJECT_STALL = 15.0
_STT_STALL = 180.0
_LANE_WATCH_INTERVAL = 2.0  # seconds between lane-health checks


def _dlog(message: str) -> None:
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')}  [engine] {message}\n")
    except OSError:
        pass


class _Lane:
    """A serial, self-healing single-worker task lane.

    Tasks run one at a time in submission order — the engine depends on that (a
    capture *start* must run before its matching *stop*; pastes must stay
    ordered). The hazard of one worker is that a single task wedged in a native
    call that never returns — a ⌘V into a hung window server, a CoreAudio open
    caught in a device-change race — jams every later task in the lane forever,
    which is precisely the "it just stops working until I restart it" failure.

    So the engine's watchdog periodically calls :meth:`recover_if_stalled`: if the
    in-flight task has run past a deadline it could only reach by hanging, the
    wedged worker is abandoned (left as a leaked daemon thread that dies with the
    process) and a fresh executor takes over, so the *next* task runs. The lane
    heals itself instead of requiring an app restart.

    :meth:`shutdown` is always bounded — it never waits unconditionally on a
    worker, so a wedged lane can't freeze the quit path.
    """

    def __init__(self, name: str, stall_seconds: float, on_recover=None) -> None:
        self.name = name
        self._stall = stall_seconds
        self._on_recover = on_recover
        self._lock = threading.Lock()
        self._closed = False
        # Monotonic time the current task began, or None when idle. Written only by
        # the active worker; a single attribute read/write is atomic in CPython, so
        # the watchdog samples it without locking.
        self._started_at: Optional[float] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = self._spawn()

    def _spawn(self) -> threading.Thread:
        # Daemon worker: an abandoned (wedged) worker after a recovery can never
        # hold up interpreter exit — unlike a ThreadPoolExecutor thread, which the
        # concurrent.futures atexit handler would try to join forever.
        t = threading.Thread(
            target=self._run, args=(self._queue,), name=self.name, daemon=True
        )
        t.start()
        return t

    def _run(self, q: "queue.Queue") -> None:
        while True:
            item = q.get()
            if item is None:  # shutdown / abandon sentinel
                return
            fn, args = item
            self._started_at = time.monotonic()
            try:
                fn(*args)
            except Exception as exc:  # a lane task must never kill its worker
                _dlog(f"lane '{self.name}' task error: {exc}")
            finally:
                self._started_at = None

    def submit(self, fn, *args) -> None:
        """Queue work on the lane (FIFO, one at a time). No-op if the lane is closed."""
        with self._lock:
            if self._closed:
                return
            self._queue.put((fn, args))

    def recover_if_stalled(self) -> bool:
        """If the in-flight task has hung past the deadline, rebuild the lane.

        Returns True if a wedged worker was abandoned and a fresh one took over.
        """
        started = self._started_at
        if started is None or (time.monotonic() - started) < self._stall:
            return False
        with self._lock:
            if self._closed:
                return False
            # Abandon the wedged worker + its queue (the stuck daemon thread dies
            # with the process) and start fresh, so new work runs immediately.
            self._queue = queue.Queue()
            self._started_at = None
            self._thread = self._spawn()
        if self._on_recover is not None:
            try:
                self._on_recover()
            except Exception:
                pass
        return True

    def shutdown(self, wait: bool = True, timeout: float = 1.0) -> None:
        """Stop the lane, never blocking longer than ``timeout``.

        Signals the worker to finish its current task and exit, then (optionally)
        joins it with a hard cap — so a wedged task can't freeze the quit path the
        way the old unbounded pool ``shutdown(wait=True)`` did. The worker is a
        daemon, so even a join that times out leaves nothing to block process exit.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
            t = self._thread
        if wait:
            t.join(timeout=timeout)


class DictationEngine:
    def __init__(self, on_state: Optional[StateCallback] = None) -> None:
        self.recorder = Recorder()
        self.transcriber = create_transcriber()
        self.feedback = Feedback()
        # Each lane is a self-healing single worker (see _Lane). A task wedged in a
        # never-returning native call gets its worker abandoned and rebuilt by the
        # lane watchdog, so one stuck operation can no longer jam its lane forever.
        self._pool = _Lane("dictate-stt", _STT_STALL)
        # A *separate* single worker that owns BOTH ends of capture — starting the
        # mic on press and finalising it on release. It must not share the
        # transcription pool: if a previous clip is still transcribing, the next
        # release's capture has to stop *now*, not wait in line behind it. And it
        # must not run on the event-tap thread: opening/rebuilding the CoreAudio
        # input stream can take tens-to-hundreds of ms (cold device, or after the
        # default mic changes), which is long enough for macOS to blow the tap's
        # servicing deadline and *disable the whole tap* — dropping that press and
        # every event after it ("it just stops working after a while"). Press and
        # release both queue here, so start always runs before its matching stop.
        # If a capture op ever hangs, recovery also resets the recorder so the next
        # press starts cleanly.
        self._capture_pool = _Lane("dictate-cap", _CAPTURE_STALL, on_recover=self._reset_recorder)
        # Text injection (clipboard write + ⌘V to the window server) runs here, OFF
        # the transcription pool. A paste can block on a wedged window server; if it
        # shared the STT worker, that one stuck paste would queue every future
        # transcription behind it forever ("stopped working"). On its own worker a
        # stuck paste only delays later pastes, never transcription, and the overlay
        # still retracts. Single worker keeps pastes (and the clipboard) ordered.
        self._inject_pool = _Lane("dictate-inj", _INJECT_STALL)
        # Watches all three lanes and rebuilds any whose in-flight task has hung,
        # so a wedged native call self-heals within a couple of seconds instead of
        # bricking dictation until the app is restarted.
        self._stop_lane_watchdog = threading.Event()
        self._lane_watchdog = threading.Thread(
            target=self._watch_lanes, name="dictate-laneguard", daemon=True
        )
        self._lane_watchdog.start()
        self._on_state: StateCallback = on_state or (lambda state, info=None: None)
        # Mic warm-window bookkeeping. The recorder leaves its stream running
        # between takes so a follow-up press captures instantly; after
        # CONFIG.mic_warm_seconds of no presses we stop it to release the mic
        # (turning off the macOS "in use" dot). _take_gen is bumped on every press
        # so a release timer that fires after a newer take simply no-ops.
        self._idle_lock = threading.Lock()
        self._idle_timer: Optional[threading.Timer] = None
        self._take_gen = 0
        # Set once shutdown() begins so the daemon idle-release Timer (which can
        # fire after the pools are gone) doesn't submit into a closed pool.
        self._shutting_down = False
        # When did the model last do real work? Used to decide whether to re-warm
        # it on the next key-down (see on_press). Starts "now" so the warmup at
        # launch counts and we don't double-warm the very first take.
        self._last_activity = time.monotonic()

    def _emit(self, state: str, info: Optional[dict] = None) -> None:
        try:
            self._on_state(state, info)
        except Exception as exc:  # a UI bug must never break dictation
            print(f"[engine] state callback error: {exc}", flush=True)

    def warmup(self) -> float:
        t0 = time.monotonic()
        self.transcriber.warmup()
        elapsed = time.monotonic() - t0
        self._emit("ready")
        return elapsed

    def prewarm_audio(self) -> None:
        """Open the mic stream ahead of the first press (best-effort, never raises).

        The cold CoreAudio device-open otherwise lands on the very first key-down.
        Building it now — before the hotkey is even installed, so no press can race
        it — means the first ``recorder.start()`` is just a cheap resume. Capture is
        not started, so the mic-in-use indicator stays off until you actually hold fn.
        """
        try:
            self.recorder.prewarm()
        except Exception as exc:  # best-effort; start() will rebuild on demand
            _dlog(f"audio prewarm skipped: {exc}")

    # -- hotkey callbacks (run on the event-tap thread; MUST return fast) -----
    # macOS disables an event tap whose callback overruns the servicing
    # deadline, dropping fn transitions. So these do the bare minimum and hand
    # any blocking work (the grace-window wait, transcription) to worker threads.
    def on_press(self) -> None:
        # Runs on the event-tap thread — it MUST return almost instantly or macOS
        # disables the tap (see _capture_pool). Emitting the UI state and the start
        # cue are non-blocking, so they stay here for instant feedback; the only
        # potentially-slow part — opening/resuming the mic stream — is handed to the
        # capture worker. The tiny hand-off latency is far below human reaction time
        # and never risks the tap.
        self._cancel_idle_release()  # a press means we're active; don't release the mic
        self._emit("listening")
        self.feedback.listening()
        self._capture_pool.submit(self._begin)

    def _begin(self) -> None:
        """Start capture off the tap thread (paired with _finalize on release)."""
        try:
            self._begin_impl()
        except Exception as exc:  # a worker task must never die or the pool wedges
            _dlog(f"begin failed: {exc}")

    def _begin_impl(self) -> None:
        self.recorder.start()
        # If the model has been idle long enough that the OS may have paged it
        # out, page it back in *now* — concurrently with you speaking — so the
        # real transcription on release runs hot. Queued on the transcription
        # pool so it serialises ahead of (and is usually finished before) the
        # real take. This is the main lever for "slow after a pause".
        idle = CONFIG.warm_after_idle
        if idle > 0 and (time.monotonic() - self._last_activity) > idle:
            self._pool.submit(self._prime)
        self._last_activity = time.monotonic()

    def _prime(self) -> None:
        """Cheap dummy inference to keep the model/Metal/pages warm. Never raises."""
        try:
            self.transcriber.warmup()
        except Exception as exc:  # priming is best-effort
            _dlog(f"prime skipped: {exc}")

    # -- mic warm-window -----------------------------------------------------
    def _cancel_idle_release(self) -> None:
        """Mark a new take and cancel any pending mic release. Called on press."""
        with self._idle_lock:
            self._take_gen += 1
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

    def _arm_idle_release(self) -> None:
        """After a take, schedule the mic to be released once the warm window of
        inactivity passes. Called on release (off the tap thread)."""
        window = CONFIG.mic_warm_seconds
        if window <= 0:
            # Warming disabled: release immediately (old behaviour — mic dot off
            # between takes, every press pays the start-up cost).
            self._capture_pool.submit(self.recorder.pause)
            return
        with self._idle_lock:
            gen = self._take_gen
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            timer = threading.Timer(window, self._fire_idle_release, args=(gen,))
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _fire_idle_release(self, gen: int) -> None:
        # Runs on the Timer thread; do the actual stream stop on the capture worker
        # so it serialises with start/stop (never races a concurrent press). A
        # submit after shutdown is a harmless no-op (the lane is closed), so the
        # _shutting_down guard is just an early-out, not a correctness requirement.
        if self._shutting_down:
            return
        self._capture_pool.submit(lambda: self._release_if_idle(gen))

    def _release_if_idle(self, gen: int) -> None:
        with self._idle_lock:
            if gen != self._take_gen:
                return  # a newer take happened since arming; stay warm
        if self.recorder._recording:
            return  # mid-take; the next release will re-arm
        self.recorder.pause()

    def on_release(self) -> None:
        # Don't touch the recorder here — recorder.stop() sleeps for the trailing
        # grace window, which must never run on the tap thread. Hand it off (to the
        # same worker as _begin, so stop always follows its matching start).
        self._capture_pool.submit(self._finalize)

    def _finalize(self) -> None:
        try:
            audio, duration = self.recorder.stop()  # includes the post-release grace
            # Stream is now warm-but-idle; schedule its release after the warm window.
            self._arm_idle_release()
            rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
            peak = float(np.abs(audio).max()) if audio.size else 0.0
            _dlog(f"release: {audio.size} samples, {duration:0.2f}s, rms={rms:0.4f}, peak={peak:0.4f}")
            if duration < CONFIG.min_record_seconds or audio.size == 0:
                self._emit("empty")
                self._emit("idle")
                return
            # Held the key but said nothing: skip the model entirely. Running
            # Whisper on a silent clip costs 1.5-5 s (it amplifies the room tone
            # and hallucinates a stock phrase we then discard) — and on an idle
            # model the real decode also waits behind the warm-up prime. The gate
            # keys off the clip's own dynamic range, so a genuine quiet word still
            # passes through; only clear silence is short-circuited.
            if not has_speech(audio, CONFIG.sample_rate):
                _dlog(f"no speech (silent clip, rms={rms:0.4f}) — skipping transcription")
                self.feedback.error()
                self._emit("empty")
                self._emit("idle")
                return
            if rms < 1e-4:
                _dlog("audio is silent — likely missing Microphone permission for this app")
            self._emit("transcribing", {"duration": duration})
            self._pool.submit(self._process, audio, duration)
        except Exception as exc:  # never let the capture worker die, never get stuck
            _dlog(f"finalize failed: {exc}")
            self._emit("idle")

    def cancel(self) -> None:
        """Abort an in-progress capture without transcribing.

        Used by the Windows hotkey when it detects a Ctrl-chord shortcut (e.g.
        Ctrl+C) rather than a genuine push-to-talk hold.
        """
        if self.recorder._recording:  # cheap best-effort; stop + discard
            self.recorder.stop()
            self._arm_idle_release()  # stream is left warm; schedule its release
        self._emit("idle")

    # -- worker thread -------------------------------------------------------
    def _process(self, audio: np.ndarray, duration: float) -> None:
        # The whole body is guarded and "idle" is emitted in `finally`, so no
        # matter what fails the worker survives and the overlay always retracts
        # (never left stuck in the "thinking" shimmer with its 60 fps timer running).
        t0 = time.monotonic()
        try:
            try:
                text = self.transcriber.transcribe(audio)
            except Exception as exc:
                self.feedback.error()
                self._emit("error", {"error": str(exc)})
                return

            if not text:
                elapsed = time.monotonic() - t0
                _dlog(f"transcribed in {elapsed:0.2f}s -> {text!r}")
                self.feedback.error()
                self._emit("empty")
                return

            # Polish is pure regex (sub-millisecond); include it in the timed window
            # so the log reflects the true time-to-paste.
            text = polish(text)
            elapsed = time.monotonic() - t0
            _dlog(f"transcribed+polished in {elapsed:0.2f}s -> {text!r}")

            self._emit("result", {"text": text, "elapsed": elapsed, "duration": duration})
            self._last_activity = time.monotonic()  # keep the warm window open
            # Deliver the text off this (STT) worker so a wedged paste can't stall
            # the next transcription; persistence is best-effort tail work.
            self._inject_pool.submit(self._deliver, text)
            self._record_history(text, duration, elapsed)
        except Exception as exc:  # defensive: keep the STT worker alive
            _dlog(f"process failed: {exc}")
            self._emit("error", {"error": str(exc)})
        finally:
            self._emit("idle")

    def _deliver(self, text: str) -> None:
        """Paste the text + play the done cue. Runs on the inject worker; guarded
        so a paste failure never kills the worker or leaks an exception."""
        try:
            inject(text)
            self.feedback.done()
        except Exception as exc:
            _dlog(f"inject failed: {exc}")

    def _record_history(self, text: str, duration: float, elapsed: float) -> None:
        """Persist a successful transcription locally (best-effort, never raises)."""
        if not CONFIG.save_history:
            return
        try:
            from .store import get_store

            get_store().add(
                text,
                duration_s=duration,
                elapsed_s=elapsed,
                model=getattr(self.transcriber, "model_name", ""),
                lang=CONFIG.language or "auto",
            )
        except Exception as exc:  # history must never break dictation
            _dlog(f"history write failed: {exc}")

    # -- lane health ---------------------------------------------------------
    def _reset_recorder(self) -> None:
        """Recovery hook after a wedged capture task is abandoned.

        The stuck task may have left the recorder mid-take; clear the flag so the
        next press starts a fresh capture instead of short-circuiting on
        ``if self._recording: return``. Plain atomic writes, safe to do from the
        watchdog thread. Best-effort."""
        try:
            self.recorder._recording = False
            self.recorder._level = 0.0
        except Exception:
            pass

    def _watch_lanes(self) -> None:
        """Rebuild any lane whose in-flight task has hung. Runs for the app's life."""
        lanes = (self._capture_pool, self._pool, self._inject_pool)
        while not self._stop_lane_watchdog.wait(_LANE_WATCH_INTERVAL):
            for lane in lanes:
                try:
                    if lane.recover_if_stalled():
                        _dlog(f"lane '{lane.name}' wedged on a hung task — rebuilt it; dictation recovered")
                except Exception as exc:  # the watchdog must never die
                    _dlog(f"lane watchdog error: {exc}")

    def shutdown(self) -> None:
        # Drain in dependency order, but with BOUNDED waits so a wedged worker can
        # never freeze the quit path (the original unbounded wait=True turned a
        # stuck paste/audio-open into a frozen, un-quittable app). Order:
        #   capture  -> finishes any start/stop/finalize (and stops queuing work),
        #   stt      -> finishes the in-flight transcription (it may hand text to
        #               the inject pool, which is still open at this point),
        #   inject   -> finishes the paste + done cue,
        #   recorder -> closed last, once no capture task can still touch it.
        # Anything still wedged past these timeouts is left to the caller's deadman
        # (menu_app force-exits the process), so quit is always prompt.
        self._shutting_down = True
        self._cancel_idle_release()
        self._stop_lane_watchdog.set()
        self._capture_pool.shutdown(wait=True, timeout=1.0)
        self._pool.shutdown(wait=True, timeout=1.0)
        self._inject_pool.shutdown(wait=True, timeout=1.0)
        self.recorder.close()
