"""Compatibility exports; source analysis and rewriting live separately."""
from .source_analysis.normalize import normalized
from .source_analysis.scopes import prepare
from .instrumentation.samples import inject

__all__ = ["normalized", "prepare", "inject"]
