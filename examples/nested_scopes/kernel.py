"""Small exact-integer fixture for source scope identity (no debug calls)."""
import tilelang.language as T


def build():
    @T.prim_func
    def nested(x: T.Tensor((37,), "int32"), y: T.Tensor((69,), "int32")):
        with T.Kernel(1, threads=32) as bx:
            tx = T.get_thread_binding()
            acc = T.alloc_var("int32")
            acc = 0
            for i in T.serial(2, 4):
                outer = i
                if tx % 2 == 0:
                    for j in T.serial(1, 3):
                        if j == 1:
                            for i in T.serial(3, 5):
                                value = tx * 1000 + j * 100 + i
                                acc = acc + value
                        elif tx % 4 == 0:
                            other = tx * 1000 + j
                            acc = acc + other
                else:
                    odd = tx + outer
                    acc = acc + odd
            y[tx] = acc
            for z in T.serial(0):
                empty = tx + z
            if False:
                inactive = tx
            for q in T.Parallel(64):
                if q < 37:
                    tail = x[q] + q * 10
                    y[32 + q] = tail
    return nested


def local_build():
    @T.prim_func
    def local_values(x: T.Tensor((32,), "int32"), y: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as bx:
            tx = T.get_thread_binding()
            local_buf = T.alloc_local((2,), "int32")
            local_buf[0] = x[tx]
            local_buf[1] = x[tx] + 7
            flag = tx % 2 == 0
            wide = T.cast(tx, "int64") + T.int64(9007199254740993)
            f16 = T.cast(tx * 0.25 - 3.5, "float16")
            bf16 = T.cast(tx * 0.25 - 3.5, "bfloat16")
            f32 = T.cast(tx * 0.25 - 3.5, "float32")
            y[tx] = local_buf[0] + local_buf[1]
    return local_values
