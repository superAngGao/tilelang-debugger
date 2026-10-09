"""Resolve the user's source input without assuming a driver sibling filename."""
import json
from pathlib import Path


def load(config_file, source_path=None):
    config_path = Path(config_file).resolve()
    original = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise ValueError("observation config must be an object")
    selected = source_path if source_path is not None else original.get("source")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("specify a nonempty --source path or config source")
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() if source_path is not None else config_path.parent) / path
    path = path.resolve()
    source = path.read_text(encoding="utf-8")
    effective = dict(original, source=str(path))
    provenance = dict(source_path=str(path), source_input=selected,
                      selected_by="cli" if source_path is not None else "config",
                      config_path=str(config_path), original_config=original,
                      worker_snapshot="source/kernel.py")
    return source, effective, provenance
