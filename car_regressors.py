import os
import threading
import atexit
import time
from collections import deque
from typing import Any, Dict, Optional, Set, Deque, TypedDict

try:
    from sklearn.ensemble import ExtraTreesRegressor
    import joblib
except Exception:
    ExtraTreesRegressor = None  # type: ignore
    joblib = None  # type: ignore

N_ESTIMS = 32

# Singleton state
_models: Optional[Dict[int, Any]] = None
_trained: Optional[Set[int]] = None
_replay: Optional[Deque[Any]] = None
_lock = threading.Lock()

def init(num_actions: int,
         *,
         n_estimators: int = N_ESTIMS,
         min_samples_leaf: int = 2,
         random_state: int = 42,
         n_jobs: int = -1,
         replay_maxlen: int = 20000) -> None:
    """Initialize one regressor per action (indexed 0..num_actions-1) and a shared replay buffer."""
    global _models, _trained, _replay
    with _lock:
        if _models is not None:
            return
        if ExtraTreesRegressor is None:
            raise RuntimeError("scikit-learn is required. Install with: py -m pip install scikit-learn")
        _models = {
            i: ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=n_jobs,
            ) for i in range(num_actions)
        }
        _trained = set()
        _replay = deque(maxlen=replay_maxlen)

def is_initialized() -> bool:
    return _models is not None

def models() -> Dict[int, Any]:
    if _models is None:
        raise RuntimeError("Shared regressors not initialized. Call init(...) or bootstrap(...) first.")
    return _models

def trained_actions() -> Set[int]:
    if _trained is None:
        raise RuntimeError("Shared regressors not initialized. Call init(...) or bootstrap(...) first.")
    return _trained

def replay() -> Deque[Any]:
    if _replay is None:
        raise RuntimeError("Shared regressors not initialized. Call init(...) or bootstrap(...) first.")
    return _replay

def reset() -> None:
    """Clear shared models and buffers (useful for restarting training)."""
    global _models, _trained, _replay
    with _lock:
        _models = None
        _trained = None
        _replay = None

class _Snapshot(TypedDict, total=False):
    num_actions: int
    models: Dict[int, Any]
    trained: Set[int]
    replay: list[Any]
    replay_maxlen: int

def save(dirpath: str) -> None:
    """Persist models, trained set, and replay buffer to a directory."""
    if _models is None or _trained is None or _replay is None:
        raise RuntimeError("Shared regressors not initialized.")
    if joblib is None:
        raise RuntimeError("joblib is required to save models. Install with: py -m pip install joblib")
    os.makedirs(dirpath, exist_ok=True)
    snap: _Snapshot = {
        "num_actions": len(_models),
        "models": _models,
        "trained": set(_trained),
        "replay": list(_replay),
        "replay_maxlen": _replay.maxlen or 20000,
    }
    joblib.dump(snap, os.path.join(dirpath, "fqi-" + str(N_ESTIMS) + "_estimators.joblib"))

def load(dirpath: str) -> None:
    """Load models, trained set, and replay buffer from a directory."""
    global _models, _trained, _replay
    if joblib is None:
        raise RuntimeError("joblib is required to load models. Install with: py -m pip install joblib")
    p = os.path.join(dirpath, "fqi.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    snap: _Snapshot = joblib.load(p)
    with _lock:
        _models = snap["models"]
        _trained = set(snap.get("trained", set()))
        maxlen = int(snap.get("replay_maxlen", 20000))
        _replay = deque(snap.get("replay", []), maxlen=maxlen)

# Autosave state
_autosave_dir: Optional[str] = None
_autosave_interval: Optional[int] = None
_autosave_thread: Optional[threading.Thread] = None

def enable_autosave(dirpath: str, interval_sec: int | None = 60) -> None:
    """Register autosave on exit and optional periodic checkpoints."""
    global _autosave_dir, _autosave_interval, _autosave_thread
    _autosave_dir = dirpath
    _autosave_interval = interval_sec

    def _save_once() -> None:
        try:
            save(_autosave_dir or dirpath)
        except Exception:
            pass

    atexit.register(_save_once)

    if interval_sec is not None and interval_sec > 0:
        if _autosave_thread is None or not _autosave_thread.is_alive():
            def _loop():
                while True:
                    time.sleep(interval_sec)
                    _save_once()
            _autosave_thread = threading.Thread(target=_loop, daemon=True, name="FQI-Autosave")
            _autosave_thread.start()

def bootstrap(dirpath: str, num_actions: int, *, autosave_interval: int = 300) -> None:
    """Load if checkpoint exists; else init fresh. Then enable autosave."""
    os.makedirs(dirpath, exist_ok=True)
    ckpt = os.path.join(dirpath, "fqi.joblib")
    if os.path.exists(ckpt):
        load(dirpath)
    else:
        init(num_actions=num_actions)
    enable_autosave(dirpath, autosave_interval)