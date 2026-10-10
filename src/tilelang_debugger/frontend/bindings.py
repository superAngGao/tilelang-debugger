"""Preserve existing object identities through eager frontend assignments."""


def reference(value):
    """Prevent binding an alias from naming the original Var/Buffer in place.

    Other expressions intentionally use normal frontend binding, preserving
    the single-evaluation behavior of the generated argument assignment.
    """
    from tvm import tirx
    from tvm.script.ir_builder.tirx import meta_var
    return meta_var(value) if isinstance(value, (tirx.Var, tirx.Buffer)) else value
