"""Assemble source, observations and offline diagnostics into one report."""
from collections import defaultdict
from datetime import datetime, timezone

from ..diagnostics.values import samples, summarize
from ..diagnostics.operations import describe, checks
from .model import portable
from .load import read


def build(data, near_zero=1e-8):
    records, accesses = defaultdict(list), defaultdict(list)
    for row in data['records']:
        records[row['launch'], row['point']].append(row)
    for row in data['accesses']:
        accesses[row['launch'], row['point']].append(row)
    comparisons = {(l['launch'], p['point']): p for l in (data['analysis'] or {}).get('launches', []) for p in l['comparisons']}
    points = []
    for contract in data['contracts']:
        launch = contract['launch']
        for p in contract['points']:
            key = launch, p['id']
            access = p.get('access', {})
            mode = p.get('mode', 'value')
            values = samples(records[key], data['comparisons']) if mode == 'value' else []
            points.append(dict(key=f'{launch}:{p["id"]}', id=p['id'], launch=launch, compile=contract['compile'], line=p['line'],
                               when=p['when'], mode=mode, buffer=p['buffer'], dtype=access.get('dtype', p.get('dtype')),
                               memory=access.get('memory_scope', p.get('scope')), shape=p.get('shape', []) if mode == 'value' else None,
                               config=p, operation=describe(data['source'], p['line']), values=values, accesses=accesses[key],
                               summary=summarize(values), comparison=comparisons.get(key),
                               coverage=data['run']['coverage'][str(launch)][p['id']]))
    for p in data['monitor']['points']:
        if p['id'] in data['run']['unlaunched_points']:
            points.append(dict(key='unlaunched:' + p['id'], id=p['id'], launch=None, compile=None, line=p['line'], when=p['when'],
                               mode=p.get('mode', 'value'), buffer=p['buffer'], dtype=None, memory=None, shape=[], config=p,
                               operation=describe(data['source'], p['line']), values=[], accesses=[], summary=summarize([]), comparison=None,
                               coverage=dict(capture_integrity='unlaunched')))
    compiler = []
    for c in data['execution']['compiles']:
        for mode in ('baseline', 'instrumented'):
            directory = data['capture'] / mode / 'compiles' / str(c['compile'])
            compiler.append(dict(compile=c['compile'], mode=mode, settings=read(directory / 'compile.json'),
                                 frontend=(directory / 'frontend.py').read_text(encoding='utf-8'),
                                 device=(directory / 'device.py').read_text(encoding='utf-8'),
                                 cuda=(directory / 'kernel.cu').read_text(encoding='utf-8')))
    diagnostics = checks(points, near_zero)
    all_values = [r for p in points for r in p['values']]
    special_points = [dict(key=p['key'], line=p['line'], launch=p['launch'], id=p['id'],
                           nan=p['summary']['nan'], posinf=p['summary']['posinf'], neginf=p['summary']['neginf'])
                      for p in sorted(points, key=lambda p: p['line']) if any(p['summary'][k] for k in ('nan', 'posinf', 'neginf'))]
    return portable(dict(schema='tilelang-debug-report-v1', generated_at=datetime.now(timezone.utc).isoformat(),
                         integer_encoding='integers outside JavaScript safe range are decimal strings; bits are hexadecimal strings',
                         run=data['run'], environment=data['environment'], monitor=data['monitor'],
                         source=dict(path=data['run']['source_path'], text=data['source']), points=points, compiler=compiler,
                         numeric_summary=summarize(all_values), operations=diagnostics, special_points=special_points,
                         analysis=dict(status='linked' if data['analysis'] else 'not_provided',
                                       explanation='已校验采集来源与实际值；展示保存的 reference 结果，未重新执行 reference。' if data['analysis'] else '未提供 reference 分析；仍可查看采样值、非有限值和访问参数。',
                                       result=data['analysis']),
                         requirements=[
                             dict(id='2.1-1', title='采集范围、等级与上下文', status='available', detail='展示实际范围、选区、预算、执行层级和 memory scope；摘要/明细是展示层级，不替代分层采集证明。'),
                             dict(id='2.1-2', title='源码与操作语义关联', status='available', detail='源码行、语句、调用及访问操作角色；不推断任意程序的数据依赖。'),
                             dict(id='2.2-numeric', title='数值偏差', status='available' if data['analysis'] else 'not_analyzed', detail='实际值、reference、误差及保存的匹配结果；未采集元素保持缺失。'),
                             dict(id='2.2-special', title='NaN / Inf 产生与传播线索', status='observed_subset', detail='按观察点展示数量和分布；源码顺序不代表运行时顺序或异常产生源。'),
                             dict(id='2.2-cast', title='类型转换精度', status='observed' if any(x['kind']=='conversion' and x['status']=='observed' for x in diagnostics) else 'not_collected', detail='同源码行的显式 scalar 转换前后按执行身份配对；变化不自动判错。'),
                             dict(id='2.2-sensitive', title='数值敏感操作', status='observed' if any(x['kind']!='conversion' and x['status']=='observed' for x in diagnostics) else 'not_collected', detail=f'已观察除数绝对值 ≤ {near_zero}、sqrt 负输入、rsqrt 非正输入及非有限值；不证明条件内操作已执行。'),
                             dict(id='2.2-access', title='索引、形状、布局与范围', status='available' if data['accesses'] else 'not_collected', detail='逻辑索引/shape/stride/mask 和边界结果；布局/硬件适配需结合 IR 人工核查。'),
                             dict(id='2.3-ir', title='运行数据与编译产物', status='available', detail='前端 IR、最终设备 IR、CUDA、编译配置及 baseline/插桩版本。'),
                             dict(id='2.3-hardware', title='硬件范围', status='project_scope', detail='本工作组验证 NVIDIA H200；其他芯片由合作组承担，此报告不声明多后端验收完成。')]))
