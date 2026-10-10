"""Build a self-contained HTML view and a portable evidence directory."""
import json
import math
from pathlib import Path
import shutil

from ..analysis import manifest, save
from .load import load
from .build import build


def html(model):
    root = Path(__file__).parent
    template = (root / 'templates/report.html').read_text(encoding='utf-8')
    style = (root / 'assets/report.css').read_text(encoding='utf-8')
    script = '\n'.join((root / 'assets' / name).read_text(encoding='utf-8') for name in ('app.js', 'source.js', 'values.js', 'accesses.js', 'compiler.js'))
    payload = json.dumps(model, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    # Insert user data last, after all template substitutions.
    return template.replace('/*STYLE*/', style).replace('/*SCRIPT*/', script).replace('/*DATA*/', payload)


def generate(capture, output, analysis=None, near_zero=1e-8):
    capture, output = Path(capture).resolve(), Path(output).resolve()
    analysis = Path(analysis).resolve() if analysis else None
    if not math.isfinite(near_zero) or near_zero < 0:
        raise ValueError('near-zero threshold must be finite and nonnegative')
    for source in (capture, analysis):
        if source and (source == output or source in output.parents or output in source.parents):
            raise ValueError('report output must be separate from capture and analysis')
        if source and any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('report evidence must not contain symbolic links')
    if output.exists():
        raise FileExistsError('report output already exists')
    data = load(capture, analysis)
    model = build(data, near_zero)
    output.mkdir(parents=True)
    try:
        shutil.copytree(capture, output / 'evidence/capture')
        if analysis:
            shutil.copytree(analysis, output / 'evidence/analysis')
        for source, copied, expected in ((capture, output / 'evidence/capture', data['capture_hashes']),
                                          (analysis, output / 'evidence/analysis', data['analysis_hashes'])):
            if source and (manifest(source) != expected or manifest(copied) != expected):
                raise ValueError('evidence changed while generating report')
        save(output / 'report.json', model)
        (output / 'report.html').write_text(html(model), encoding='utf-8')
        save(output / 'manifest.json', dict(schema='tilelang-debug-report-manifest-v1', status='completed',
                                           run_id=data['run']['run_id'], files=manifest(output),
                                           generator='tilelang-debugger/report-v1', analysis_status=model['analysis']['status']))
    except Exception as exc:
        save(output / 'manifest.json', dict(status='failed', error=str(exc)))
        raise
    return dict(status='completed', output=str(output), report=str(output / 'report.html'), points=len(model['points']))
