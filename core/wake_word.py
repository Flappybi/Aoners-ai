"""Local wake phrase detection for A-ONERS using offline Vosk recognition.

Vosk runs offline after its small English model has been downloaded once. The
recognized transcript is checked for phonetic spellings of "Hey A-ONERS".
"""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

WAKE_MODEL = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 0.5  # Kept for callers that used the former detector API.


def _model_cache_dir() -> Path:
    """Keep the Vosk model in a predictable, user-writable location."""
    return Path.home() / ".cache" / "vosk"


def _prepare_model_path() -> Path:
    # Vosk reads this variable while it is imported. Respect an explicit
    # operator setting, otherwise use Vosk's standard user cache directory.
    path = Path(os.environ.setdefault("VOSK_MODEL_PATH", str(_model_cache_dir())))
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_installed() -> bool:
    """True if the Vosk runtime is importable without loading the model."""
    try:
        return importlib.util.find_spec("vosk") is not None
    except Exception:
        return False


def is_ready() -> bool:
    """True when the Vosk package and English model files are present."""
    if not is_installed():
        return False
    try:
        configured = os.environ.get("VOSK_MODEL_PATH")
        roots = [Path(configured)] if configured else [_model_cache_dir()]
        roots.extend((Path.home() / "AppData" / "Local" / "vosk",
                      Path.home() / ".cache" / "vosk",
                      Path("/usr/share/vosk")))
        for root in roots:
            model_dir = root / WAKE_MODEL
            if ((model_dir / "am").is_dir()
                    and (model_dir / "conf").is_dir()
                    and (model_dir / "graph").is_dir()):
                return True
    except Exception:
        pass
    return False


def install_and_download(logger: Callable[[str], None] = print,
                         notify: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Install Vosk if needed, then download and load the English wake model."""
    _tell = notify or (lambda _msg: None)
    try:
        _prepare_model_path()
        if not is_installed():
            logger("Wake word: installing Vosk (one-time)…")
            _tell("Wake word: installing Vosk (one-time)…")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "vosk"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or [""]
                return False, f"Vosk install failed: {tail[0][:160]}"

        logger("Wake word: downloading the A-ONERS English model (about 40 MB)…")
        _tell("Wake word: downloading the A-ONERS English model (about 40 MB)…")
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        model = Model(model_name=WAKE_MODEL)
        del model

        if not is_ready():
            return False, "Vosk installed, but the English wake model is not ready."
        logger("Wake word: ready for 'Hey A-ONERS'.")
        return True, "A-ONERS wake word installed and ready."
    except Exception as exc:
        return False, f"wake model setup failed: {exc}"


class WakeWordDetector:
    """Run local phrase recognition on a dedicated thread."""

    def __init__(self, on_detect: Callable[[], None],
                 threshold: float = DEFAULT_THRESHOLD,
                 logger: Callable[[str], None] = print,
                 notify: Callable[[str], None] | None = None):
        self._on_detect = on_detect
        self._threshold = threshold
        self._logger = logger
        self._notify = notify or (lambda _msg: None)
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._thread: threading.Thread | None = None
        self._running = False
        self._model = None
        self._recognizer = None
        self._ready = False

    def start(self) -> bool:
        """Load the local model and start inference. Safe to call repeatedly."""
        if self._running:
            return True
        try:
            _prepare_model_path()
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            self._model = Model(model_name=WAKE_MODEL)
            self._recognizer = KaldiRecognizer(self._model, SAMPLE_RATE)
        except Exception as exc:
            self._logger(f"Wake word: could not load the A-ONERS model — {exc}")
            self._notify("A-ONERS wake word unavailable — use the WAKE NOW button.")
            self._model = None
            self._recognizer = None
            return False

        self._running = True
        self._ready = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="WakeWordThread",
        )
        self._thread.start()
        self._logger("Wake word: listening for 'Hey A-ONERS'.")
        return True

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._recognizer = None
        self._model = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def feed(self, frame_int16) -> None:
        """Cheap non-blocking queue push from the real-time microphone callback."""
        if not self._running:
            return
        try:
            data = frame_int16[:, 0].copy() if getattr(frame_int16, "ndim", 1) > 1 else frame_int16.copy()
            self._queue.put_nowait(data)
        except queue.Full:
            pass
        except Exception:
            pass

    @staticmethod
    def _matches_wake_phrase(text: str) -> bool:
        """Accept only recognized variants of "Hey A-ONERS"."""
        words = "".join(char for char in text.lower() if char.isalnum() or char.isspace()).split()
        if not words or words[0] != "hey":
            return False
        following = words[1:]
        # The intended Indonesian-friendly pronunciation is often spoken as
        # "Hey Ewaners". Vosk may keep that as one word instead of splitting
        # it into "A owners"; accept those direct phonetic spellings too.
        if following and following[0] in {
            "winner", "winners", "wonder", "wonders",
            "ewaner", "ewaners", "ewanner", "ewanners",
        }:
            return True
        has_a = any(word in {"a", "ay"} for word in following[:2])
        has_oners = any(word in {"oner", "oners", "one", "ones", "owner", "owners",
                                 "wonder", "wonders"}
                        for word in following[1:5])
        return has_a and has_oners

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self._queue.get()
                if frame is None or not self._running:
                    break
                audio = frame.tobytes()
                if self._recognizer.AcceptWaveform(audio):
                    text = json.loads(self._recognizer.Result()).get("text", "")
                else:
                    text = json.loads(self._recognizer.PartialResult()).get("partial", "")

                if self._matches_wake_phrase(text):
                    self._drain()
                    self._recognizer.Reset()
                    try:
                        self._on_detect()
                    except Exception as exc:
                        self._logger(f"Wake word: on_detect error — {exc}")
            except Exception as exc:
                self._logger(f"Wake word: inference error — {exc}")

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
