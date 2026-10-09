"""Validate the unchanged packed host launch and derive descriptor fields.

No FFI callbacks, opaque descriptor reads, or invented constructor return codes.
"""
from pathlib import Path
from .access_ir import evaluate, pure
from .instrument import digest


SOURCE_PINS = {
    "src/cuda/runtime.cc": "b5e1651e3cb699ec2cef52885f027d4fd156c4729915cfa69dd8b78f773cb72f",
    "src/cuda/transform/lower_hopper_intrin.cc": "f3536bd6b2abc560e4ef7867feb43c2fe708ba1cd438e626795bfed1c3d25ac7",
    "tilelang/cuda/pipeline.py": "ce0284f410110a24988047818fabec6099ae0e6c8d7e0367abf279a803a6e169",
}
SHAPES = {"gelu": [[4096],[4096]], "sum": [[4,257],[4]], "sum_unpadded": [[4,256],[4]],
          "gemm": [[128,256],[256,256],[128,256]],
          "gqa": [[1,256,2,64],[1,384,1,64],[1,384,1,64],[1,256,2,64]]}


class HostBindings:
    def __init__(self,module,prim_func,out_idx,case):
        import tvm
        from tvm import tirx as T
        self.case=case
        self.out_idx=[out_idx] if isinstance(out_idx,int) else list(out_idx)
        self.prim_func=prim_func
        if self.out_idx != [len(SHAPES[case])-1] or len(prim_func.params)!=len(SHAPES[case]):
            raise ValueError("original packed tensor argument contract differs")
        self.roles=[prim_func.buffer_map[v].name for v in prim_func.params]
        self.tensors={}
        self.binds={}
        self.descriptors=[]
        self.evidence=dict(tensor_roles=self.roles,descriptor_provenance="derived_from_host_ir_and_launch_bindings")
        hosts=[f for f in module.functions.values() if int(f.attrs.get("calling_conv",0))!=2]
        devices=[(gv,f) for gv,f in module.functions.items() if int(f.attrs.get("calling_conv",0))==2]
        if len(hosts)!=1 or len(devices)!=1:
            raise ValueError("one host and one device function required")
        host=hosts[0]
        if len(host.params)!=4 or host.params[1].name!="args":raise ValueError("unknown packed host ABI")
        self.packed_args=host.params[1]
        gv,device=devices[0]
        def collect(n):
            if isinstance(n,T.Call) and n.op.name not in {"tirx.tvm_struct_get","tirx.isnullptr","tirx.if_then_else","tirx.tvm_call_packed","tirx.handle_add_byte_offset","tirx.tvm_stack_alloca","tirx.ret"}:
                raise ValueError("unknown host call can invalidate descriptor binding")
            if isinstance(n,T.Bind):
                if n.var in self.binds:
                    raise ValueError("duplicate host binding")
                self.binds[n.var]=n.value
        T.stmt_functor.post_order_visit(host.body,collect)
        calls=[]
        def visit(n,path=()):
            if isinstance(n,T.SeqStmt):
                for i,child in enumerate(n.seq):visit(child,path+((n,i),))
            elif isinstance(n,T.AttrStmt):
                visit(n.body,path)
            elif isinstance(n,T.Evaluate) and isinstance(n.value,T.Call) and n.value.op.name=="tirx.tvm_call_packed":
                if n.value.args[0].value not in (gv.name_hint,"__tvm_tensormap_create_tiled","__tvm_set_device"):
                    raise ValueError("unknown host packed call")
                calls.append((n.value,path))
            elif isinstance(n,(T.Bind,T.DeclBuffer,T.AssertStmt)):
                pass
            elif isinstance(n,T.Evaluate) and isinstance(n.value,T.Call) and n.value.op.name=="tirx.ret" and int(n.value.args[0])==0:
                pass
            else:
                raise ValueError(f"unreviewed host control flow/statement: {type(n).__name__}")
        visit(host.body)
        launches=[(c,p) for c,p in calls if c.args[0].value==gv.name_hint]
        if len(launches)!=1:
            raise ValueError("host must call target device exactly once")
        launch,launch_path=launches[0]
        self.evidence["device_parameter_order"]=[v.name for v in device.params]
        self.launch_scalars=[]
        self.launch_dims=list(launch.args[1+len(device.params):])
        # Verify global data parameters by actual packed tensor identity, not their order or names.
        for i,v in enumerate(device.params):
            if str(v.dtype)=="handle" and v.type_annotation.storage_scope=="global":
                packed=self.pointer_index(launch.args[1+i])
                if self.roles[packed] != v.name:
                    raise ValueError("device data role does not match original PrimFunc/host binding")
            elif str(v.dtype)!="handle":
                arg=launch.args[1+i]
                if case!="gqa" or v.name not in ("seq_len_q","seq_len_kv") or not isinstance(arg,T.Var) or arg.name!=v.name:
                    raise ValueError("device dynamic scalar has wrong host argument")
                self.launch_scalars.append(arg)
        metadata=host.attrs.get("tma_descriptor_args",{})
        constructors=[(c,p) for c,p in calls if c.args[0].value=="__tvm_tensormap_create_tiled"]
        expected_count=2 if case=="gemm" else 4 if case=="gqa" else 0
        if len(metadata)!=expected_count or len(constructors)!=expected_count:
            raise ValueError("descriptor constructor count differs")
        for desc,arguments in metadata.items():
            allocation=self.binds[desc]
            if not isinstance(allocation,T.Call) or allocation.op.name!="tirx.tvm_stack_alloca" or allocation.args[0].value!="tvm_ffi_any" or int(allocation.args[1])!=16:
                raise ValueError("descriptor must have one original stack allocation")
            # Descriptor storage may appear only in its constructor and unique device call.
            uses=[]
            def descriptor_use(n):
                if isinstance(n,T.Call) and any(isinstance(v,T.Var) and v.same_as(desc) for v in n.args):uses.append(n)
            T.stmt_functor.post_order_visit(host.body,descriptor_use)
            if len(uses)!=2 or any(n.op.name!="tirx.tvm_call_packed" for n in uses):
                raise ValueError("descriptor has unreviewed uses/alias writes")
            matches=[(c,p) for c,p in constructors if c.args[1].same_as(desc)]
            if len(matches)!=1 or not tvm.ir.structural_equal(list(matches[0][0].args),list(arguments)):
                raise ValueError("descriptor metadata differs from actual host constructor")
            call,path=matches[0]
            # A shared sequential ancestor with earlier child proves constructor dominates launch.
            if not any(a.same_as(b) and i<j for a,i in path for b,j in launch_path):
                raise ValueError("constructor does not dominate launch")
            positions=[i for i,arg in enumerate(launch.args[1:1+len(device.params)]) if isinstance(arg,T.Var) and arg.same_as(desc)]
            if len(positions)!=1 or device.params[positions[0]].name!=desc.name:
                raise ValueError("descriptor device argument position mismatch")
            dtype,rank=int(arguments[2]),int(arguments[3])
            if dtype!=6 or rank!=(2 if case=="gemm" else 4) or len(arguments)!=9+4*rank:
                raise ValueError("unreviewed descriptor layout")
            index=self.pointer_index(arguments[4])
            if desc.name!=self.roles[index]+"_desc":
                raise ValueError("descriptor role differs from original tensor")
            self.descriptors.append(dict(name=desc.name,index=index,rank=rank,args=list(arguments[5:]),position=positions[0]))
        if metadata:
            import tilelang
            root=Path(tilelang.__file__).resolve().parent.parent
            actual={name:digest((root/name).read_bytes()) for name in SOURCE_PINS}
            if actual!=SOURCE_PINS:
                raise ValueError("unreviewed descriptor runtime/lowering source revision")
            self.evidence["source_pins"]=actual
        # No hidden constructor/call in a Bind/Assert, or descriptor writes via aliases.
        actual_calls=[]
        T.stmt_functor.post_order_visit(host.body,lambda n:actual_calls.append(n) if isinstance(n,T.Call) and n.op.name=="tirx.tvm_call_packed" else None)
        if len(actual_calls)!=len(calls) or any(not any(c.same_as(k) for k,_ in calls) for c in actual_calls):
            raise ValueError("hidden host call outside reviewed statement path")
        self.evidence.update(constructors=expected_count,unique_launch=True,constructor_dominance=True)

    def handle_index(self,var):
        from tvm import tirx as T
        e=self.binds[var]
        if not isinstance(e,T.Select):raise ValueError("unknown packed handle binding")
        false=e.false_value
        true=e.true_value
        if not (isinstance(false,T.Call) and false.op.name=="tirx.tvm_struct_get" and int(false.args[2])==15
                and isinstance(true,T.Call) and true.op.name=="tirx.handle_add_byte_offset" and int(true.args[1])==24):
            raise ValueError("unknown packed tensor representation")
        import tvm
        if not tvm.ir.structural_equal(true.args[0],false):raise ValueError("packed alternatives differ")
        if not false.args[0].same_as(self.packed_args):raise ValueError("packed handle has wrong args root")
        index=int(false.args[1])
        condition=e.condition
        if not isinstance(condition,T.EQ) or not isinstance(condition.a,T.Var) or not isinstance(condition.b,T.IntImm) or int(condition.b)!=70:
            raise ValueError("packed representation discriminator differs")
        type_index=self.binds[condition.a]
        if not isinstance(type_index,T.Call) or type_index.op.name!="tirx.tvm_struct_get" or not type_index.args[0].same_as(self.packed_args) or int(type_index.args[1])!=index or int(type_index.args[2])!=13:
            raise ValueError("packed type discriminator uses wrong tensor")
        return int(false.args[1])

    def pointer_index(self,var):
        from tvm import tirx as T
        e=self.binds[var]
        if not isinstance(e,T.Call) or e.op.name!="tirx.tvm_struct_get" or int(e.args[1])!=0 or int(e.args[2])!=1:
            raise ValueError("global data pointer is not a tensor binding")
        index=self.handle_index(e.args[0])
        if not 0<=index<len(self.roles):raise ValueError("invalid packed tensor index")
        return index

    def check_tensor(self,index,tensor):
        import torch
        if not isinstance(tensor,torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous() or list(tensor.shape)!=SHAPES[self.case][index]:
            raise ValueError("actual tensor shape/stride outside reviewed contract")
        dtype=torch.bfloat16 if self.case=="gelu" else torch.float16
        if tensor.dtype!=dtype:raise ValueError("tensor dtype differs")
        self.tensors[index]=tensor

    def bind_inputs(self,inputs):
        indices=[i for i in range(len(self.roles)) if i not in self.out_idx]
        if len(inputs)!=len(indices):raise ValueError("input tensor count differs")
        for i,t in zip(indices,inputs):self.check_tensor(i,t)
        for v in self.launch_scalars:
            if self.scalar(v)!=(256 if v.name=="seq_len_q" else 384):raise ValueError("actual device scalar differs")
        dims=[self.scalar(e) for e in self.launch_dims]
        expected={"gelu":[2,128,1,1],"sum":[2,128,1,1,512],"sum_unpadded":[2,128,1,1,1536],
                  "gemm":[2,1,256,1,1,98304],"gqa":[2,2,1,384,1,1,98304]}[self.case]
        if dims!=expected:raise ValueError(f"actual host launch dimensions differ: {dims}")

    def scalar(self,expr):
        from tvm import tirx as T
        import tvm
        mapping={}
        variables=[]
        T.stmt_functor.post_order_visit(expr,lambda n:variables.append(n) if isinstance(n,T.Var) else None)
        for v in variables:
            binding=self.binds[v]
            if not isinstance(binding,T.Cast) or not isinstance(binding.value,T.BufferLoad):
                raise ValueError("unknown dynamic descriptor scalar")
            load=binding.value
            origin=self.binds[load.buffer.data]
            if not isinstance(origin,T.Call) or origin.op.name!="tirx.tvm_struct_get" or int(origin.args[2])!=2 or len(load.indices)!=1:
                raise ValueError("dynamic scalar is not actual tensor shape")
            index=self.handle_index(origin.args[0]);axis=int(load.indices[0])
            if self.case!="gqa" or (v.name,index,axis) not in (("seq_len_q",0,1),("seq_len_kv",1,1)) or str(binding.dtype)!="int32":
                raise ValueError("unreviewed scalar shape/cast binding")
            # Evaluate the original Cast as well as subsequent descriptor arithmetic.
            replaced=T.Cast(binding.dtype,T.IntImm(load.dtype,int(self.tensors[index].shape[axis])))
            mapping[v]=evaluate(replaced,{})
        pure(expr)
        return evaluate(expr,mapping)

    def finish(self,out):
        import torch
        outputs=[out] if isinstance(out,torch.Tensor) else list(out)
        if len(outputs)!=len(self.out_idx):raise ValueError("output tensor count differs")
        for i,t in zip(self.out_idx,outputs):self.check_tensor(i,t)
        result=[]
        for d in self.descriptors:
            r=d["rank"];tensor=self.tensors[d["index"]]
            values=[self.scalar(e) for e in d["args"]]
            if self.case=="gemm":
                expected=[256,128 if d["index"]==0 else 256,2,512,64,128,1,1,0,3,2,0]
                modes=[1,0]
            else:
                seq,heads=(256,2) if d["index"] in (0,3) else (384,1)
                expected=[64,seq,heads,1,2,heads*128,128,seq*heads*128,64,64 if heads==2 else 128,1,1,1,1,1,1,0,3,2,0]
                modes=[3,1,2,0]
            if values!=expected:
                raise ValueError(f"descriptor configuration differs: {d['name']} {values}")
            result.append(dict(name=d["name"],tensor_role=self.roles[d["index"]],packed_tensor_index=d["index"],device_parameter_index=d["position"],
                               provenance="derived_from_host_ir_and_launch_bindings",rank=r,dtype_enum=6,
                               tensor_shape=list(tensor.shape),tensor_stride=list(tensor.stride()),tensor_data_ptr=tensor.data_ptr(),
                               global_dim=values[:r],global_stride_raw=values[r:2*r],cuda_global_strides=values[r+1:2*r],
                               box_dim=values[2*r:3*r],element_strides=values[3*r:4*r],settings=values[4*r:],
                               mode_to_tensor_axis=modes,host_expressions=[str(e) for e in d["args"]],
                               construction_evidence="unchanged dominating constructor, fatal-on-error pinned runtime, successful launch; outer process must also exit zero"))
        return result
