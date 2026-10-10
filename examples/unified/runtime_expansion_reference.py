def reference(inputs, points):
    x = inputs[0]
    n = x.numel()
    expected = x + 3
    values = {}
    for p in points:
        if p['buffer'] == 'y':
            samples = [dict(coordinates=[], ordinals=[], index=i, value=int(expected[i])) for i in range(n)]
        else:
            samples = [dict(coordinates=[i], ordinals=[0], index=0, value=int(expected[i])) for i in range(n)]
        values[p['id']] = dict(schema=3, key='logical', dtype='int32', samples=samples, atol=0, rtol=0)
    return dict(points=values, arguments_after=[dict(tensor=x.clone(), atol=0, rtol=0)], outputs=[dict(tensor=expected, atol=0, rtol=0)])
