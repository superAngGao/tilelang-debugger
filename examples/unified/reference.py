"""Exact independent intermediate and post-launch reference for kernel.py."""
import torch


def tensor(value):
    return dict(tensor=value, atol=0, rtol=0)


def reference(inputs, points):
    x, n = inputs
    point = points[0]
    case = point['id']
    output = x.clone()
    samples = []
    def add(thread, visit, coords, ordinals, index, value):
        samples.append(dict(thread=thread, visit=visit, coordinates=coords, ordinals=ordinals, index=index, value=int(value)))
    if case == 'parallel':
        for i in range(61):
            add(i % 32, i // 32, [i], [0], 0, x[i] + i)
        output[:61] += torch.arange(61, dtype=x.dtype)
    elif case in {'pipeline', 'group', 'dynamic', 'while', 'negative'}:
        threads = range(128, 256) if case == 'group' else range(32)
        for tx in threads:
            iterations = {'pipeline': range(2, 7), 'group': range(3), 'dynamic': range(tx % n), 'while': (1, 3), 'negative': range(8, 1, -2)}[case]
            for visit, k in enumerate(iterations):
                ordinal = k if case == 'while' else visit + 1
                add(tx, visit, [k], [ordinal], 0, x[tx] + k)
                output[tx] = x[tx] + k
    elif case == 'two_dim':
        for tx in range(32):
            add(tx, 0, [], [], 0, x[tx])
            add(tx, 0, [], [], 1, tx + 17)
        output[:32] += torch.arange(32, dtype=x.dtype) + 17
    elif case in {'fragment', 'shared', 'global'}:
        for i in range(0, 32, 2):
            add(0, 0, [], [], i, x[i] + (3 if case == 'fragment' else 7 if case == 'global' else 0))
        if case == 'fragment':
            output[:64] += 3
        if case == 'global':
            output[:32] += 7
    elif case == 'inplace':
        for tx in range(32):
            add(tx, 0, [], [], 0, x[tx] + n)
        output += n
    spec = dict(schema=3, key='execution', dtype='int32', samples=samples, atol=0, rtol=0)
    return dict(points={point['id']: spec}, arguments_after=[tensor(output if case == 'inplace' else x), n],
                outputs=[] if case == 'inplace' else [tensor(output)])
