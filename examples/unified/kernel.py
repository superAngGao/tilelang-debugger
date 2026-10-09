"""Ordinary kernels for the unified capture matrix; no debugger calls."""
import tilelang.language as T


def build(case):
    threads = 256 if case == 'group' else (16, 2) if case == 'two_dim' else 32

    @T.prim_func
    def main(x: T.Tensor((256,), 'int32'), y: T.Tensor((256,), 'int32'), n: T.int32):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding(0) + T.get_thread_binding(1) * 16
            for q in T.Parallel(256):
                y[q] = x[q]
            if case == 'parallel':
                for i in T.Parallel(65):
                    if i < 61:
                        parallel_value = x[i] + i
                        y[i] = parallel_value
            elif case == 'pipeline':
                for k in T.Pipelined(2, stop=7, num_stages=3):
                    pipeline_value = x[tx] + k
                    y[tx] = pipeline_value
            elif case == 'group':
                with T.ws(1):
                    for k in T.serial(3):
                        group_value = x[tx] + k
                        y[tx] = group_value
            elif case == 'two_dim':
                local = T.alloc_local((2,), 'int32')
                local[0] = x[tx]
                local[1] = tx + 17
                y[tx] = local[0] + local[1]
            elif case == 'dynamic':
                for k in T.serial(tx % n):
                    dynamic_value = x[tx] + k
                    y[tx] = dynamic_value
            elif case == 'while':
                k = T.alloc_var('int32', init=0)
                while k < n:
                    k = k + 1
                    if k == 2:
                        continue
                    if k == 4:
                        break
                    while_value = x[tx] + k
                    y[tx] = while_value
            elif case == 'negative':
                for k in T.serial(8, 1, step=-2):
                    negative_value = x[tx] + k
                    y[tx] = negative_value
            elif case == 'fragment':
                fragment = T.alloc_fragment((64,), 'int32')
                for i in T.Parallel(64):
                    fragment[i] = x[i] + 3
                T.copy(fragment, y[0:64])
            elif case == 'shared':
                shared = T.alloc_shared((64,), 'int32')
                T.copy(x[0:64], shared)
                T.sync_threads()
                T.copy(shared, y[0:64])
            elif case == 'global':
                T.sync_threads()
                y[tx] = y[tx] + 7
    return main


def inplace_build():
    @T.prim_func
    def inplace(x: T.Tensor((32,), 'int32'), delta: T.int32):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            updated = x[tx] + delta
            x[tx] = updated
    return inplace
