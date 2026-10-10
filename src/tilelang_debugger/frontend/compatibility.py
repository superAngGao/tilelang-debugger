"""Behavioral frontend capability probes; no source-hash admission list."""
import importlib
import inspect


class FrontendCompatibilityError(RuntimeError):
    pass


def check():
    """Probe required frontend APIs without lowering or a CUDA context."""
    import tilelang
    result = dict(schema=1, tilelang=tilelang.__version__, tilelang_path=tilelang.__file__,
                  status='compatible_interfaces', checks=[])
    current = 'frontend imports'
    try:
        from tvm import tirx
        from tvm.script.ir_builder.tirx import meta_var
        from tilelang.language.kernel import KernelLaunchFrame
        from tilelang.language.copy_op import _normalize_copy_regions
        from .access import region
        current = 'compile interception'
        for function in (tilelang.compile, importlib.import_module('tilelang.jit').compile):
            if 'out_idx' not in inspect.signature(function).parameters:
                raise ValueError('compile has no out_idx parameter')
        result['checks'].append(current)
        current = 'kernel geometry'
        for name in ('Current', 'get_thread_extents', 'get_thread_bindings', 'get_block_bindings'):
            if not callable(getattr(KernelLaunchFrame, name, None)):
                raise ValueError('KernelLaunchFrame.' + name + ' is unavailable')
        result['checks'].append(current)
        current = 'identity-preserving binding'
        if not callable(meta_var):
            raise ValueError('meta_var unavailable')
        result['checks'].append('meta_var available; identity behavior covered by frontend contract tests')
        current = 'copy region normalization'
        src = tirx.decl_buffer((4, 49), 'int32', name='compat_source')
        dst = tirx.decl_buffer((1, 32), 'int32', name='compat_destination', scope='shared.dyn')
        pair = _normalize_copy_regions(src[1, 32], dst)
        expected = ((src, [1, 32]), (dst, [0, 0]))
        for value, (buffer, start) in zip(pair, expected, strict=True):
            actual, origin, extent = region(value)
            if not actual.same_as(buffer) or [int(v) for v in origin] != start or [int(v) for v in extent] != [1, 32]:
                raise ValueError('copy normalization changed region semantics')
        result['region_op'] = str(pair[0].op.name)
        result['checks'].append(current)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise FrontendCompatibilityError(f'TileLang {result["tilelang"]}: incompatible {current}: {exc}') from exc
    return result
