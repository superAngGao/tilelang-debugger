"""Logical snapshot bytes mapped onto the original strided storage."""
import itertools
import math


def byte_spans(shape, stride, offset, width):
    if not math.prod(shape):
        return
    block, rank = 1, len(shape)
    while rank and (shape[rank - 1] == 1 or stride[rank - 1] == block):
        rank -= 1
        block *= shape[rank]
    for coordinates in itertools.product(*(range(n) for n in shape[:rank])):
        start = (offset + sum(i * s for i, s in zip(coordinates, stride))) * width
        yield start, block * width


def storage_end(shape, stride, offset, width):
    return (offset + (1 + sum((n - 1) * s for n, s in zip(shape, stride)) if math.prod(shape) else 0)) * width


def restore_storage(items, folder):
    """Validate overlapping bytes and return shared bytearrays per alias group."""
    from ..protocols.dtypes import UNIFIED_WIDTH
    groups, masks = {}, {}
    for item in items:
        if item['kind'] != 'tensor':
            continue
        group, size = item['alias_group'], item['storage_bytes']
        if group not in groups:
            groups[group], masks[group] = bytearray(size), bytearray(size)
        target, mask = groups[group], masks[group]
        raw = (folder / item['file']).read_bytes()
        cursor = 0
        for start, length in byte_spans(item['shape'], item['stride'], item['storage_offset'], max(1, UNIFIED_WIDTH[item['dtype']] // 8)):
            chunk = raw[cursor:cursor + length]
            if any(mask[start:start + length]):
                if any(flag and old != new for flag, old, new in zip(mask[start:start + length], target[start:start + length], chunk)):
                    raise ValueError('overlapping tensor snapshots contain inconsistent bytes')
            target[start:start + length] = chunk
            mask[start:start + length] = b'\1' * length
            cursor += length
        if cursor != len(raw):
            raise ValueError('logical snapshot size differs from view')
    return groups
