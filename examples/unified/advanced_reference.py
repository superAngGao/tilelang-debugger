import torch


def tensor(value):
    return dict(tensor=value, atol=0, rtol=0)


def reference(inputs, points):
    x = inputs[0]
    if not points:
        return dict(points={}, arguments_after=[tensor(x + 1)], outputs=[])
    p = points[0]
    case = p['id']
    samples = []
    output = x.clone()
    def add(tx, visit, coords, ordinals, index, val):
        samples.append(dict(thread=tx, visit=visit, coordinates=coords, ordinals=ordinals, index=index, value=val))
    if case == 'alias':
        for tx in range(32):
            add(tx, 0, [], [], 0, int(x[tx] + inputs[2]))
        updated = x + inputs[2]
        after, outputs = [tensor(updated), tensor(updated), tensor(inputs[2])], []
    else:
        if case == 'group_fragment':
            for i in range(128):
                add(p['reader'], 0, [], [], i, int(x[i]) + 11)
            output[:128] += 11
        elif case == 'pipeline_fragment':
            for k in range(4):
                for i in range(32):
                    add(0, k, [k], [k + 1], i, int(x[i]) + k)
            output[:32] += 3
        elif case == 'nested':
            for tx in range(0, 32, 2):
                visit = 0
                for outer in range(2):
                    for inner in range(outer + 1):
                        selected = p['thread'] is None or p['thread'] == tx
                        for s in p['scopes']:
                            if 'selection' in s:
                                ordinal = outer + 1 if s['names'] == ['outer'] else inner + 1
                                selected &= ordinal == s['selection']['iteration']
                        if selected:
                            add(tx, visit, [outer, inner], [outer + 1, inner + 1], 0, int(x[tx]) + outer * 10 + inner)
                            visit += 1
            output[:32:2] += 11
        elif case == 'repeated':
            for tx in range(32):
                for k in range(3):
                    add(tx, k, [k + 1], [k + 1], 0, int(x[tx]))
        elif case == 'final_return':
            for tx in range(32):
                add(tx, 0, [], [], 0, int(x[tx]) + 1)
            output[:32] += 1
        elif case == 'types':
            dtype, scope = p['dtype'], p['scope']
            def cast(v):
                return torch.tensor(v, dtype=getattr(torch, dtype)).item()
            if scope == 'scalar':
                for tx in range(32):
                    val = tx + (9223372036854775809 if dtype == 'uint64' else 9007199254740993 if dtype == 'int64' else 1.0000000000000002 if dtype == 'float64' else 3)
                    add(tx, 0, [], [], 0, tx % 2 == 0 if dtype == 'bool' else cast(val))
            elif scope == 'local':
                for tx in range(32):
                    for i in range(2):
                        add(tx, 0, [], [], i, (tx + i) % 2 == 0 if dtype == 'bool' else cast(tx + 3 + i))
            else:
                for i in range(32):
                    add(0, 0, [], [], i, i % 2 == 0 if dtype == 'bool' else cast(i + 3))
        after, outputs = [tensor(x)], [tensor(output)]
    return dict(points={p['id']: dict(schema=3, key='execution', dtype=p['dtype'], samples=samples, atol=0, rtol=0)},
                arguments_after=after, outputs=outputs)
