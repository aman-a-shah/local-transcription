"""Local push-to-talk dictation for macOS (hold fn, speak, release)."""

__all__ = ["DictationApp", "main"]
__version__ = "1.0.2"


def __getattr__(name):
    """Import the app lazily.

    Pulling in ``.app`` eagerly drags the whole UI/audio stack — and its native
    deps (sounddevice, pyobjc/pywin32) — into *any* ``import dictate.<submodule>``.
    The test environment installs only the lightweight deps, so we defer that
    import until ``DictationApp``/``main`` is actually accessed, keeping
    ``dictate.polish``, ``dictate.core``, and ``dictate.transcriber`` importable
    on their own.
    """
    if name in ("DictationApp", "main"):
        from .app import DictationApp, main

        globals().update(DictationApp=DictationApp, main=main)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
