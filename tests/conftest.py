import importlib.util
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE))


def load_stage(filename: str):
    """Import a numbered stage module (names starting with a digit can't be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        filename.removesuffix(".py").lstrip("0123456789_"), PIPELINE / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
