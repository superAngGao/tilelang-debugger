"""Strict event coverage and offline interpretation, independent of log ordering."""
import json
import re


def join_words(lo,hi,signed):
    if not 0 <= lo < 2**32 or not 0 <= hi < 2**32:
        raise ValueError("invalid uint32 word")
    value=lo+(hi<<32)
    return value-(1<<64) if signed and value >= 1<<63 else value


def parse(text,points,run_id):
    bysite={p["site"]:p for p in points}
    expected={}
    for p in points:
        for k in p["expected_keys"]:
            key=(p["site"],k["thread"],*k["loops"],k["lane"])
            if key in expected:
                raise ValueError("duplicate manifest identity")
            expected[key]=k["active"]
    seen,records=set(),[]
    for line in text.splitlines():
        if "TLACC" not in line:
            continue
        parts=line.split("|")
        if len(parts)!=25 or parts[0]!="TLACC1" or parts[1] not in bysite or parts[2]!="0" or any(not re.fullmatch(r"-?\d+",s) for s in parts[3:]):
            raise ValueError("malformed/unknown/truncated access record")
        p=bysite[parts[1]]
        values=list(map(int,parts[3:]))
        bx,by,bz,tx,*tail=values
        loop_values=tail[:4]
        lane,active=tail[4:6]
        if [bx,by,bz]!=p["block"] or active not in (0,1) or any(not -(1<<31)<=v<(1<<31) for v in values[:10]):
            raise ValueError("record domain/ABI mismatch")
        elected=p["elected"]
        if elected and not elected[0]<=tx<elected[1]:
            raise ValueError("TMA issuer outside elected warp")
        key=(p["site"],None if elected else tx,*loop_values,lane)
        if key not in expected or key in seen or expected[key]!=active:
            raise ValueError("unexpected/duplicate key or wrong access mask")
        seen.add(key)
        operands=[join_words(values[10+i*2],values[11+i*2],p["signed"][i]) for i in range(6)]
        if not elected and operands[1:]!=[0]*5:
            raise ValueError("nonzero reserved operands")
        record=dict(run_id=run_id,launch=0,site=p["site"],point=p["id"],block=[bx,by,bz],thread=tx,warp_lane=tx%32,
                    loops=loop_values,vector_lane=lane,active=bool(active),operands=operands,provenance="runtime_ir_operands")
        if not elected:
            record.update(element_offset=operands[0],byte_offset=operands[0]*2)
            # Current reviewed source buffers are all 16-bit.
            record["bounds"]="masked" if not active else "in_bounds" if 0<=operands[0]<p["shape"][0] else "out_of_bounds"
        else:
            if p["rank"]==2 and operands[3:5]!=[0,0] or p["kind"]=="tma_store" and operands[5]!=-1:
                raise ValueError("invalid reserved TMA operands")
            record.update(shared_element_offset=operands[0],coordinates=operands[1:1+p["rank"]],barrier_index=None if p["kind"]=="tma_store" else operands[5])
        records.append(record)
    if seen!=set(expected):
        raise ValueError(f"incomplete access trace: {len(seen)}/{len(expected)} events")
    return sorted(records,key=lambda r:(r["site"],r["loops"],r["thread"],r["vector_lane"]))


def validate_workers(folder,sanitizer):
    for mode in ("baseline","instrumented"):
        p=folder/mode
        process=json.loads((p/"process.json").read_text())
        state=json.loads((p/"pipeline-state.json").read_text())
        if process!={"returncode":0,"timeout":False} or state!={"pipeline_calls":1,"codegen_calls":1,"restored":True}:
            raise ValueError("worker/pipeline did not complete cleanly")
        if not json.loads((p/"launch-gate.json").read_text()).get("passed"):
            raise ValueError("launch gate did not pass")
        if sanitizer:
            log=(p/f"{sanitizer}.log").read_text()
            summary="RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if sanitizer=="racecheck" else "ERROR SUMMARY: 0 errors"
            if log.count(summary)!=1 or re.search(r"(?:ERROR SUMMARY:|RACECHECK SUMMARY:) [1-9]",log):
                raise ValueError("sanitizer did not report zero errors")
            command=json.loads((p/"command.json").read_text())
            if command[:5]!=["compute-sanitizer","--tool",sanitizer,"--error-exitcode","86"]:
                raise ValueError("sanitizer command differs from requested validation")


def report(records,points,descriptors):
    result=dict(complete=True,records=len(records),points=[],descriptors=descriptors,
                limitations=["IR operands do not prove transaction completion or physical memory traffic", "TMA regions are derived; physical shared swizzle is not expanded"])
    lines=["# Access trace", "", "完整采集；索引来自 GPU 上原访问表达式。记录不证明内存请求完成。", "", "| 源码行 | 操作 | 记录 | active | masked |", "| --- | --- | --- | --- | --- |"]
    for p in points:
        rs=[r for r in records if r["site"]==p["site"]]
        item=dict(site=p["site"],line=p["line"],operation=p["operation"],records=len(rs),active=sum(r["active"] for r in rs),masked=sum(not r["active"] for r in rs),out_of_bounds=sum(r.get("bounds")=="out_of_bounds" for r in rs))
        result["points"].append(item)
        lines.append(f"| {p['line']} | {p['operation']} {p['buffer']} | {len(rs)} | {item['active']} | {item['masked']} |")
        if p["operation"]=="transfer":
            matches=[d for d in descriptors if d["name"]==p["descriptor"]]
            if len(matches)!=1:raise ValueError("missing/ambiguous descriptor binding")
            d=matches[0]
            if d["rank"]!=p["rank"] or d["settings"]!=[0,3,2,0]:raise ValueError("unreviewed descriptor settings")
            r=rs[0]
            region=[[c,c+b] for c,b in zip(r["coordinates"],d["box_dim"])]
            inside=all(0<=lo and hi<=n for (lo,hi),n in zip(region,d["global_dim"]))
            classification=("invalid_store_origin" if p["kind"]=="tma_store" and any(c<0 for c in r["coordinates"])
                    else "in_bounds" if inside else "load_oob_zero_fill" if p["kind"]=="tma_load" else "store_oob_discard")
            logical=[None]*d["rank"]
            for bounds,axis in zip(region,d["mode_to_tensor_axis"]):logical[axis]=bounds
            item.update(descriptor=d["name"],region_in_descriptor_modes=region,region_in_tensor_axes=logical,
                        region_provenance="derived_from_runtime_operands_and_host_descriptor_bindings",
                        global_bounds=classification,
                        physical_shared_swizzle="not_expanded",shared_element_offset=r["shared_element_offset"],
                        shared_extent=p["shared_extent"],shared_alias_bytes=p["shared_alias_bytes"],barrier_index=r["barrier_index"])
    lines += ["", "TMA descriptor 参数来自真实 host IR 与本次 tensor 绑定的推导；坐标、shared 偏移、执行线程及 barrier 下标来自 GPU 记录。", "",
              "| TMA 源码行 | 原 tensor 区间（左闭右开，推导） | shared 元素偏移 | 边界解释 |", "| --- | --- | --- | --- |"] if descriptors else []
    for item in result["points"]:
        if "descriptor" in item:
            lines.append(f"| {item['line']} | {item['region_in_tensor_axes']} | {item['shared_element_offset']} | {item['global_bounds']} |")
    lines += ["", "详见 `access-points.json` 中原索引/guard/loop 表达式、`instrumented/codegen.py`、`instrumented/kernel.cu`。", ""]
    return result,"\n".join(lines)+"\n"
