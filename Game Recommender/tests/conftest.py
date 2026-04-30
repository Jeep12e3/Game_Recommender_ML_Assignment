from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _identity_cache(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


streamlit_stub = types.SimpleNamespace(cache_resource=_identity_cache)
sys.modules.setdefault("streamlit", streamlit_stub)
