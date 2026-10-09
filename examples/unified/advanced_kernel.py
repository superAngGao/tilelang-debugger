import tilelang.language as T


def build(case):
    threads = 256 if case == 'group_fragment' else 32
    @T.prim_func
    def main(x: T.Tensor((256,), 'int32'), y: T.Tensor((256,), 'int32')):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            for q in T.Parallel(256):
                y[q] = x[q]
            T.sync_threads()
            if case == 'group_fragment':
                with T.ws(1):
                    fragment = T.alloc_fragment((128,), 'int32')
                    for i in T.Parallel(128):
                        fragment[i] = x[i] + 11
                    T.copy(fragment, y[0:128])
            elif case == 'pipeline_fragment':
                fragment = T.alloc_fragment((32,), 'int32')
                for k in T.Pipelined(4, num_stages=3):
                    for i in T.Parallel(32):
                        fragment[i] = x[i] + k
                    T.copy(fragment, y[0:32])
            elif case == 'nested':
                for outer in T.serial(2):
                    for inner in T.serial(outer + 1):
                        if tx % 2 == 0:
                            nested_value = x[tx] + outer * 10 + inner
                            y[tx] = nested_value
            elif case == 'repeated':
                k = T.alloc_var('int32', init=0)
                while k < 3:
                    repeated_value = x[tx]
                    y[tx] = repeated_value
                    k = k + 1
            elif case == 'final_return':
                final_value = x[tx] + 1
                y[tx] = final_value
            return
    return main


def typed_build(dtype, scope):
    @T.prim_func
    def main(x: T.Tensor((32,), 'int32'), y: T.Tensor((32,), 'int32')):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if scope == 'scalar':
                if dtype == 'uint64':
                    typed_value = T.cast(tx, dtype) + T.uint64(9223372036854775809)
                elif dtype == 'int64':
                    typed_value = T.cast(tx, dtype) + T.int64(9007199254740993)
                elif dtype == 'float64':
                    typed_value = T.cast(tx, dtype) + T.float64(1.0000000000000002)
                else:
                    typed_value = (tx % 2 == 0) if dtype == 'bool' else T.cast(tx + 3, dtype)
                y[tx] = x[tx]
            elif scope == 'fragment':
                typed_buffer = T.alloc_fragment((32,), dtype)
                for i in T.Parallel(32):
                    typed_buffer[i] = (i % 2 == 0) if dtype == 'bool' else T.cast(i + 3, dtype)
                y[tx] = x[tx]
            elif scope == 'shared':
                typed_buffer = T.alloc_shared((32,), dtype)
                typed_buffer[tx] = (tx % 2 == 0) if dtype == 'bool' else T.cast(tx + 3, dtype)
                T.sync_threads()
                y[tx] = x[tx]
            else:
                typed_buffer = T.alloc_local((2,), dtype)
                typed_buffer[0] = (tx % 2 == 0) if dtype == 'bool' else T.cast(tx + 3, dtype)
                typed_buffer[1] = (tx % 2 == 1) if dtype == 'bool' else T.cast(tx + 4, dtype)
                y[tx] = x[tx]
    return main


def helper_build():
    @T.prim_func
    def helper(x: T.Tensor((32,), 'int32')):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            x[tx] = x[tx] + 1
    return helper


def alias_build():
    @T.prim_func
    def main(x: T.Tensor((32,), 'int32'), other: T.Tensor((32,), 'int32'), scalar: T.Tensor((1,), 'int32')):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            alias_value = x[tx] + scalar[0]
            other[tx] = alias_value
    return main
