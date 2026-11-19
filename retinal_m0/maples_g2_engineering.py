"""G2-only bounded engineering primitives. No CUDA/model construction on import."""
from __future__ import annotations
import contextlib, hashlib, json, math, os, tempfile, time
from pathlib import Path

GLOBAL_CAP = 400
TRACK_CAPS = {"R0_LAMBDA0": 200, "B1_CONT": 50, "B1_SPLIT": 50,
              "B2_CONT": 50, "B2_SPLIT": 50}
AUDIT_STEPS = {1, 5, 10, 25, 50}
RETRIES_PER_UPDATE = 8
ATTEMPTS_PER_UPDATE = RETRIES_PER_UPDATE + 1

class GateError(RuntimeError): pass

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def _pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise GateError("duplicate JSON key: "+k)
        out[k]=v
    return out

def read_json(path):
    def bad(x): raise GateError("nonfinite JSON constant: "+x)
    return json.loads(Path(path).read_text(encoding="utf-8"),object_pairs_hook=_pairs,parse_constant=bad)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+".",dir=p.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(obj,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
        d=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

@contextlib.contextmanager
def lock(path):
    import fcntl
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a+") as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise GateError("busy G2 accounting lock") from e
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def empty_ledger(binding_sha):
    return {"schema":1,"policy":"P1_B1_B2_G2_ENGINEERING_EXCEPTION_01_20261021",
      "binding_sha256":binding_sha,"global_cap":GLOBAL_CAP,"track_caps":TRACK_CAPS,
      "tracks":{},"charged_events":0,"verified_successes":0,"attempts":0,
      "amp_retries":0,"failed_events":0,"uncertain_events":0,"optimizer_events":0,
      "status":"READY"}

class Budget:
    """Write-ahead charging: unknown/failed optimizer calls are never refunded."""
    def __init__(self,path,binding_sha):
        self.path=Path(path);self.lock_path=Path(str(path)+".lock");self.binding_sha=binding_sha
    def _read(self):
        if not self.path.is_file():raise GateError("G2 ledger missing; refusing reset")
        s=read_json(self.path)
        if s.get("schema")!=1 or s.get("binding_sha256")!=self.binding_sha or s.get("global_cap")!=400 or s.get("track_caps")!=TRACK_CAPS:
            raise GateError("G2 ledger schema/binding/budget mismatch")
        totals={k:0 for k in ("charged_events","verified_successes","attempts","amp_retries","failed_events","uncertain_events","optimizer_events")}
        if any(type(s.get(k))is not int or s[k]<0 for k in totals):raise GateError('invalid aggregate counter type/value')
        for name,t in s["tracks"].items():
            if name not in TRACK_CAPS:raise GateError("unknown G2 track")
            if any(type(t.get(k))is not int or t[k]<0 for k in totals):raise GateError('invalid track counter type/value')
            if t['charged_events']!=t['verified_successes']+t['failed_events']+t['uncertain_events'] or t['optimizer_events']!=t['verified_successes']+t['failed_events']:raise GateError('inconsistent charged/event totals')
            for k in totals:totals[k]+=int(t.get(k,0))
            if t["charged_events"]>TRACK_CAPS[name]:raise GateError("per-track charged cap exceeded")
        if any(s[k]!=v for k,v in totals.items()):raise GateError("G2 aggregate counter mismatch")
        if s["charged_events"]>400:raise GateError("global charged cap exceeded")
        if sum(t.get('pending') is not None for t in s['tracks'].values())>1:raise GateError('multiple unresolved optimizer invocations')
        return s
    def initialize_isolated(self):
        with lock(self.lock_path):
            if self.path.exists():raise GateError("ledger already exists; no implicit reset")
            atomic_json(self.path,empty_ledger(self.binding_sha))
    def snapshot(self):
        with lock(self.lock_path):return self._read()
    def _mutate(self,fn):
        with lock(self.lock_path):
            s=self._read();fn(s);atomic_json(self.path,s);self._read()
    def reserve(self,track,planned_step,attempt):
        def f(s):
            if s["status"]!="READY":raise GateError("ledger is not runnable")
            if track not in TRACK_CAPS:raise GateError("unknown track")
            t=s["tracks"].setdefault(track,{"charged_events":0,"verified_successes":0,"attempts":0,"amp_retries":0,"failed_events":0,"uncertain_events":0,"optimizer_events":0,"pending":None,"retry_cursor":None})
            if any(v.get("pending") is not None for v in s["tracks"].values()):raise GateError("unresolved prior optimizer invocation")
            if s.get("require_compute") and s.get("compute_pending")!={"track":track,"step":planned_step,"attempt":attempt}:raise GateError("optimizer requires its counted forward")
            if t["charged_events"]>=TRACK_CAPS[track] or s["charged_events"]>=400:raise GateError("G2 budget exhausted")
            if type(planned_step)is not int or type(attempt)is not int or attempt<1 or attempt>9:raise GateError("invalid planned step/attempt")
            cursor=t.get("retry_cursor")
            if cursor is not None and (planned_step!=cursor["planned_step"] or attempt!=cursor["next_attempt"]):raise GateError("AMP retry must repeat the identical planned draw with next attempt number")
            if cursor is None and attempt!=1:raise GateError("new planned draw must start at attempt 1")
            if cursor is None and planned_step!=t["verified_successes"]+1:raise GateError("planned updates must be unique and strictly ordered")
            t["attempts"]+=1;s["attempts"]+=1
            t["pending"]={"planned_step":planned_step,"attempt":attempt,"reserved_at":time.time(),"outcome":"UNKNOWN"}
            atomic_json(self.path,s) # durable write-ahead before optimizer.step
        self._mutate(f)
    def resolve(self,track,outcome,*,amp_retry=False):
        if outcome not in ("success","failed","uncertain","amp_overflow"):raise GateError("bad optimizer outcome")
        def f(s):
            t=s["tracks"].get(track)
            if not t or t["pending"] is None:raise GateError("no write-ahead optimizer reservation")
            p=t["pending"];t["pending"]=None
            s["compute_pending"]=None
            if outcome=="success":
                t["retry_cursor"]=None
                t["charged_events"]+=1;t["verified_successes"]+=1;t["optimizer_events"]+=1
                for k in ("charged_events","verified_successes","optimizer_events"):s[k]+=1
            elif outcome=="failed":
                t["retry_cursor"]=None
                t["charged_events"]+=1;t["failed_events"]+=1;t["optimizer_events"]+=1
                for k in ("charged_events","failed_events","optimizer_events"):s[k]+=1
                s["status"]="FAILED_STOP"
            elif outcome=="uncertain":
                t["retry_cursor"]=None
                t["charged_events"]+=1;t["uncertain_events"]+=1
                s["charged_events"]+=1;s["uncertain_events"]+=1;s["status"]="UNCERTAIN_STOP"
            elif outcome=="amp_overflow":
                if not amp_retry:raise GateError("AMP overflow must be explicitly recorded as retry")
                # No optimizer invocation and no charged update. Enforce same draw on next attempt.
                if p["attempt"]>=9:
                    t["retry_cursor"]=None;s["status"]="FAILED_STOP"
                else:
                    t["retry_cursor"]={"planned_step":p["planned_step"],"next_attempt":p["attempt"]+1}
                t["amp_retries"]+=1;s["amp_retries"]+=1
            if t["charged_events"]>TRACK_CAPS[track] or s["charged_events"]>400:raise GateError("budget exceeded")
        self._mutate(f)
    def mark_crash_uncertain(self,track):
        self.resolve(track,"uncertain")

    def stop(self,reason):
        def f(s):
            s["status"]="FAILED_STOP";s["stop_reason"]=str(reason)
        self._mutate(f)

    def before_forward(self,track,step,attempt):
        def f(s):
            if s["status"]!="READY" or s.get("compute_pending") is not None or any(t.get("pending") for t in s["tracks"].values()):raise GateError("unresolved compute/update or stopped ledger")
            t=s["tracks"].get(track,{"verified_successes":0,"charged_events":0,"retry_cursor":None})
            if track not in TRACK_CAPS or type(step)is not int or type(attempt)is not int or not 1<=attempt<=9:raise GateError("invalid compute identity")
            if step!=t["verified_successes"]+1 or t["charged_events"]>=TRACK_CAPS[track] or s["charged_events"]>=400:raise GateError("compute exceeds remaining budget")
            cursor=t.get("retry_cursor")
            if (cursor is None and attempt!=1) or (cursor is not None and cursor!={"planned_step":step,"next_attempt":attempt}):raise GateError("compute retry drift")
            if s.get("forward_calls",0)>=3600:raise GateError("forward budget exhausted")
            s["require_compute"]=True;s["compute_pending"]={"track":track,"step":step,"attempt":attempt}
            s["forward_calls"]=s.get("forward_calls",0)+1;s["current_gradient_calls"]=0
        self._mutate(f)

    def before_gradient(self,track,step,attempt):
        def f(s):
            if s["status"]!="READY" or s.get("compute_pending")!={"track":track,"step":step,"attempt":attempt}:raise GateError("gradient without counted forward")
            limit=3 if track!="R0_LAMBDA0" and step in AUDIT_STEPS else 1
            if s.get("current_gradient_calls",0)>=limit or s.get("gradient_calls",0)>=3960:raise GateError("gradient budget exhausted")
            s["current_gradient_calls"]=s.get("current_gradient_calls",0)+1;s["gradient_calls"]=s.get("gradient_calls",0)+1
        self._mutate(f)


def optimizer_update(budget,track,planned_step,attempt,optimizer,scaler,*,amp_enabled,validate_after=None):
    """One write-ahead optimizer attempt, counted by the real post-step event."""
    budget.reserve(track,planned_step,attempt)
    observed=[]
    hook=optimizer.register_step_post_hook(lambda *_:observed.append("post"))
    try:
        if amp_enabled:
            scaler.step(optimizer);scaler.update()
        else:
            optimizer.step()
        if len(observed)==1:
            require_finite_tensors(optimizer.state_dict())
            for group in optimizer.param_groups:
                for p in group["params"]:require_finite_tensors(p)
            if validate_after is not None:validate_after()
            budget.resolve(track,"success")
            return {"outcome":"success","draw_advance":True}
        if len(observed)==0 and amp_enabled:
            budget.resolve(track,"amp_overflow",amp_retry=True)
            return {"outcome":"amp_overflow","draw_advance":False,"retry_same_draw":True,"stop":budget.snapshot()["status"]!="READY"}
        budget.resolve(track,"uncertain")
        raise GateError("optimizer post-hook event count is not exactly one")
    except BaseException:
        s=budget.snapshot()
        if s["tracks"][track].get("pending") is not None:
            budget.resolve(track,"failed" if len(observed)==1 else "uncertain")
        raise
    finally:
        hook.remove()

def require_finite_tensors(value):
    import torch
    if isinstance(value,torch.Tensor):
        if not bool(torch.isfinite(value).all()):raise GateError("nonfinite tensor state")
    elif isinstance(value,dict):
        for v in value.values():require_finite_tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value:require_finite_tensors(v)
    elif isinstance(value,float) and not math.isfinite(value):raise GateError("nonfinite scalar state")

def _tree_finite(x,path="metrics"):
    if x is None or type(x) in (str,bool,int):return
    if type(x)is float:
        if not math.isfinite(x):raise GateError("nonfinite at "+path)
        return
    if isinstance(x,dict):
        for k,v in x.items():_tree_finite(v,path+"."+str(k))
    elif isinstance(x,(list,tuple)):
        for i,v in enumerate(x):_tree_finite(v,path+f"[{i}]")
    else:raise GateError("unsupported value at "+path)

def _tensor_digest(model):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def _rng_snapshot(device=None):
    import random, numpy as np, torch
    s={"python":random.getstate(),"numpy":np.random.get_state(),"cpu":torch.get_rng_state().clone()}
    if device is not None and str(device).startswith("cuda"):s["cuda"]=torch.cuda.get_rng_state(device).clone()
    return s

def _rng_equal(a,b):
    import numpy as np, torch
    if a["python"]!=b["python"] or a["cpu"].equal(b["cpu"]) is False:return False
    x,y=a["numpy"],b["numpy"]
    return x[0]==y[0] and np.array_equal(x[1],y[1]) and x[2:]==y[2:] and (("cuda" not in a and "cuda" not in b) or ("cuda" in a and "cuda" in b and torch.equal(a["cuda"],b["cuda"])))

def same_forward_gradient_audit(model, logits, base_loss, weighted_aux_loss, *, scaler, audit_step, before_gradient=lambda:None):
    """Two same-graph autograd.grad probes; verifies .grad/RNG/scaler/model untouched."""
    import torch
    if audit_step not in AUDIT_STEPS:raise GateError("gradient audit outside frozen steps")
    params=tuple(p for p in model.parameters() if p.requires_grad)
    before_rng=_rng_snapshot(logits.device);before_model=_tensor_digest(model)
    before_scale=scaler.state_dict();before_grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
    scale=float(scaler.get_scale())
    def get(loss,retain):
        before_gradient()
        gs=torch.autograd.grad(scaler.scale(loss),params,retain_graph=retain,allow_unused=False)
        return tuple(g.detach().float()/scale for g in gs)
    gb=get(base_loss,True);ga=get(weighted_aux_loss,True)
    n0=math.sqrt(sum(float(g.double().square().sum()) for g in gb))
    n1=math.sqrt(sum(float(g.double().square().sum()) for g in ga))
    dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(gb,ga))
    cos=dot/(n0*n1) if n0 and n1 else None
    ratio=(n1/n0) if n0 else None
    out={"audit_step":audit_step,"base_grad_l2":n0,"weighted_aux_grad_l2":n1,
      "aux_to_base_l2":ratio,"cosine":cos,"zero_base_positive_aux":bool(n0==0 and n1>0),
      "pause_ratio_gt_2":bool((ratio is not None and ratio>2) or (n0==0 and n1>0)),
      "negative_cosine":bool(cos is not None and cos<0),"unscaled":True,"autograd_calls":2}
    _tree_finite(out)
    if _tensor_digest(model)!=before_model or scaler.state_dict()!=before_scale:raise GateError("gradient audit changed model or scaler")
    after_grads=[None if p.grad is None else p.grad.detach() for p in params]
    if any((x is None)!=(y is None) or (x is not None and not torch.equal(x,y)) for x,y in zip(before_grads,after_grads)):raise GateError("gradient audit changed p.grad")
    if not _rng_equal(before_rng,_rng_snapshot(logits.device)):raise GateError("gradient audit changed RNG")
    return out

def checkpoint_bundle(model,optimizer,scaler,*,track,step,next_draw,loader_generator=None):
    import random,numpy as np,torch
    state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
      "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_cpu_rng":torch.get_rng_state(),
      "torch_cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
      "track":track,"step":step,"next_draw":next_draw,
      "loader_generator_state":None if loader_generator is None else loader_generator.get_state(),
      "model_digest":_tensor_digest(model)}
    return state
def restore_bundle(bundle,model,optimizer,scaler,*,track,expected_step,loader_generator=None):
    import random,numpy as np,torch
    if bundle.get("track")!=track or bundle.get("step")!=expected_step:raise GateError("checkpoint track/step mismatch")
    if "next_draw" not in bundle or "python_rng" not in bundle or "numpy_rng" not in bundle or "torch_cpu_rng" not in bundle:raise GateError("checkpoint missing exact RNG/draw state")
    if loader_generator is not None and bundle.get("loader_generator_state") is None:raise GateError("private loader RNG missing")
    model.load_state_dict(bundle["model"],strict=True);optimizer.load_state_dict(bundle["optimizer"]);scaler.load_state_dict(bundle["scaler"])
    random.setstate(bundle["python_rng"]);np.random.set_state(bundle["numpy_rng"]);torch.set_rng_state(bundle["torch_cpu_rng"])
    if bundle.get("torch_cuda_rng") is not None:
        if not torch.cuda.is_initialized():raise GateError("CUDA RNG restore requires authorized initialized device")
        torch.cuda.set_rng_state_all(bundle["torch_cuda_rng"])
    if loader_generator is not None:loader_generator.set_state(bundle["loader_generator_state"])
    if _tensor_digest(model)!=bundle.get("model_digest"):raise GateError("restored model digest mismatch")
    return int(bundle["next_draw"])

