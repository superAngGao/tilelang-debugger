"""Compatibility dispatch for source plans, rewriting and package loading."""
from .source_analysis.legacy import readonly_syntax
from .runtime.source_loader import source_loader


def prepare(source, config):
    if config.get("schema") == 3:
        from .source_analysis.control_flow import prepare as prepare_source
    elif config.get("schema") == 2:
        from .source_analysis.scopes import prepare as prepare_source
    else:
        from .source_analysis.legacy import prepare as prepare_source
    return prepare_source(source, config)


def inject(source, points):
    if points[0].get("schema") == 3:
        from .instrumentation.unified import inject as insert
    elif points[0].get("schema") == 2:
        from .instrumentation.samples import inject as insert
    else:
        from .instrumentation.legacy import inject as insert
    return insert(source, points)
