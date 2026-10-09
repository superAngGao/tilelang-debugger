"""Selected-file import hook preserving actual package origins and restoration."""
from contextlib import contextmanager
import importlib.abc
import importlib.machinery
import linecache
from pathlib import Path
import sys
from ..instrument import Unsupported


@contextmanager
def source_loader(path, source):
    """Keep the real module origin/package; bypass pyc for the selected file only."""
    path = Path(path).resolve()
    previous_cache = dict(linecache.cache)
    previous_meta = list(sys.meta_path)
    state = {"loads": 0}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve() == path:
            raise Unsupported("selected source was imported before the source hook")

    class Loader(importlib.machinery.SourceFileLoader):
        def get_source(self, fullname):
            return source

        def get_code(self, fullname):
            state["loads"] += 1
            return self.source_to_code(source, str(path))

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, package_path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, package_path)
            if spec is not None and spec.origin and Path(spec.origin).resolve() == path:
                if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                    raise Unsupported("selected module needs a normal Python source loader")
                spec.loader = Loader(fullname, str(path))
                return spec
            return None

    linecache.cache[str(path)] = (len(source), None, source.splitlines(True), str(path))
    sys.meta_path.insert(0, Finder())
    try:
        yield state
    finally:
        sys.meta_path[:] = previous_meta
        linecache.cache.clear()
        linecache.cache.update(previous_cache)
        state["restored"] = sys.meta_path == previous_meta and linecache.cache == previous_cache
