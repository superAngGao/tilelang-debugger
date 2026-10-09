"""Protocol bit widths; safe to import without a GPU runtime."""
SAMPLE_WIDTH = {"bool": 1, "int32": 32, "int64": 64, "float16": 16, "bfloat16": 16, "float32": 32}
UNIFIED_WIDTH = {'bool': 1, 'int8': 8, 'uint8': 8, 'int16': 16, 'uint16': 16,
                 'int32': 32, 'uint32': 32, 'int64': 64, 'uint64': 64,
                 'float16': 16, 'bfloat16': 16, 'float32': 32, 'float64': 64}
