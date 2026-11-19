"""Prepare the approved engineering-only branch. No CUDA, no release or training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',nnUNet_compile='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import sys,json,datetime,shutil,ast,subprocess,tempfile,copy
from pathlib import Path
from unittest.mock import patch
from common import *

def record(title,body,authority_value=None):
    backup=ROOT/'docs/.continuity_backups'/ (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_ds_engineering')
    backup.mkdir()
    with exclusive_lock(CODE/'.deep_supervision_status_write.lock'):
        for name in ['STATUS.md','项目全局状态与研究决策.md']:
            p=CODE/name;old=p.read_text();(backup/name).write_text(old)
            first,tail=old.split('\n',1)
            new=first+'\n\n### '+title+'\n\n'+body+'\n\n'+tail
            if authority_value is not None and name=='STATUS.md':
                a,z=MARKERS
                if a in new:raise RuntimeError('existing authority must be reconciled')
                new+='\n'+a+'\n'+json.dumps(authority_value,ensure_ascii=False)+'\n'+z+'\n'
            temp=p.with_name(p.name+'.ds_record_pending')
            with temp.open('x') as f:f.write(new);f.flush();os.fsync(f.fileno())
            if p.read_text()!=old:raise RuntimeError('concurrent authority change')
            os.replace(temp,p)
    return str(backup)

def main():
    cpu=STAGE/'cpu';cpu.mkdir(exist_ok=False)
    out=STAGE/'engineering/final';out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(cpu/'matplotlib')
    fixture_path=BASE/'multiscale_capacity_ablation_v1_20261027/cpu/fixed_batches.json'
    manifest=read(fixture_path)
    q={'phase':PHASE,'seed':2027,'modes':['final'],'GPU':GPU,'official_Test_allowed':False,'science_authorized':False,
       'authorization':'User approved bounded GPU engineering: 可以，开始吧; reconfirmed after NAS recovery: 继续，我连接上了',
       'engineering':{'attempt_cap_per_mode':32,'diagnostic_forward_cap_per_mode':4,'wall_cap_seconds_per_mode':1800,'memory_cap_MiB':10240,'continuous':16,'split':[8,8]},
       'weights':[1,0,0,0,0,0,0,0],'train_ids':manifest['Train_ids'],'baseline':str(N0),'fixture_manifest':str(fixture_path),
       'stop':'any AMP skip/nonfinite/OOM/source drift/recovery mismatch; all failure/unknown costs retained; no automatic retry or science',
       'input':'fixed existing first16 Train batches; full native1024x1536 batch2; no Dev/Test',
       'initializer':'original seed2027; retain all8 heads and target pyramid',
       'scope':'engineering only; no scientific protocol freeze or 100epochs release'}
    atomic(STAGE/'ENGINEERING_PROTOCOL.json',q,new=True)
    a={'phase':PHASE,'state':'CPU_ENTRY_PRECHECK','protocol_sha256':sha(STAGE/'ENGINEERING_PROTOCOL.json'),'engineering_released':False,'science_released':False,'official_Test_allowed':False}
    backup=record('深监督消融：用户批准有界GPU工程验证，CPU入口预检中（2026-10-28）',
      '本次“可以，开始吧”及NAS恢复后“继续”批准此前4次零更新诊断+32次更新尝试、累计30分钟/10GiB的工程验证。Sol直接实施，GPU0优先；不含100epochs科学训练。先完成来源绑定、拒绝/阶段/预算/恢复入口CPU检查，再由Sol独立发布工程release。原86项CPU证据复用，原初始化/8头/CE+Dice不变，只有监督权重为[1,0,0,0,0,0,0,0]。NAS曾不可读但现已恢复，旧失败读取未产生GPU工作。当前科学HOLD；Test封存，旧结论与人工0记录保持。证据 experiments/nnunet2d_branch_20261020/deep_supervision_ablation_20261028/。',a)
    atomic(STAGE/'AUTHORIZATION.json',{'user_scope':q['authorization'],'scope':q['engineering'],'science':False,'backup':backup},new=True)
    checks={}
    def check(k,v):
        checks[k]=bool(v)
        if not v:raise AssertionError(k)
    def denied(k,fn):
        try:fn()
        except (RuntimeError,ValueError,FileNotFoundError,IndexError):check(k,True)
        else:check(k,False)
    import common as cm
    from engineering import Counter
    check('preimport_CUDA_hidden',os.environ['CUDA_VISIBLE_DEVICES']=='')
    for p in BRANCH.glob('*.py'):ast.parse(p.read_text())
    denied('production_GPU_denied_without_release',lambda:cm.validate('final','engineering'))
    denied('science_role_rejected',lambda:cm.validate('final','science'))
    denied('old_variant_rejected',lambda:cm.validate('local','engineering'))
    denied('direct_worker_parent_rejected',lambda:cm.admit_worker('final','engineering','diag',-123,'x'*48))
    # Exercise admission with isolated JSON fixtures, without importing torch or opening CUDA.
    with tempfile.TemporaryDirectory(dir=cpu) as td:
        st=Path(td);(st/'cpu').mkdir();src=st/'runtime.py';src.write_text('# fixture')
        for name,val in [('ENGINEERING_PROTOCOL.json',q),('cpu/checks.json',{'passed':True})]:atomic(st/name,val)
        bd={'protocol_sha256':sha(st/'ENGINEERING_PROTOCOL.json'),'runtime_sources':{str(src):sha(src)},'all_sources':{},'fixture_sources':{}}
        bd['binding_sha256']=canonical(bd);atomic(st/'source_binding.json',bd)
        cr={'status':'CPU_ACCEPTED','binding_sha256':bd['binding_sha256'],'checks_sha256':sha(st/'cpu/checks.json')};atomic(st/'CPU_REVIEW.json',cr)
        rr={'phase':PHASE,'role':'engineering','binding_sha256':bd['binding_sha256'],'protocol_sha256':sha(st/'ENGINEERING_PROTOCOL.json'),'CPU_review_sha256':sha(st/'CPU_REVIEW.json'),'modes':['final'],'GPU':GPU,'official_Test_allowed':False};atomic(st/'ENGINEERING_RELEASE.json',rr)
        aa={'phase':PHASE,'protocol_sha256':sha(st/'ENGINEERING_PROTOCOL.json'),'engineering_released':True,'engineering_release_sha256':sha(st/'ENGINEERING_RELEASE.json')}
        with patch.object(cm,'STAGE',st),patch.object(cm,'authority',lambda:aa):
            check('isolated_valid_admission',bool(cm.validate('final','engineering',True)))
            aa['engineering_released']=False;denied('HOLD_denied',lambda:cm.validate('final','engineering'));aa['engineering_released']=True
            aa['engineering_release_sha256']='wrong';denied('STATUS_release_digest_denied',lambda:cm.validate('final','engineering'));aa['engineering_release_sha256']=sha(st/'ENGINEERING_RELEASE.json')
            rr['phase']='OTHER';atomic(st/'ENGINEERING_RELEASE.json',rr);aa['engineering_release_sha256']=sha(st/'ENGINEERING_RELEASE.json');denied('release_phase_denied',lambda:cm.validate('final','engineering'));rr['phase']=PHASE;atomic(st/'ENGINEERING_RELEASE.json',rr);aa['engineering_release_sha256']=sha(st/'ENGINEERING_RELEASE.json')
            src.write_text('# drift');denied('runtime_drift_denied',lambda:cm.validate('final','engineering'));src.write_text('# fixture')
            cr['binding_sha256']='wrong';atomic(st/'CPU_REVIEW.json',cr);denied('CPU_binding_denied',lambda:cm.validate('final','engineering'))
        counter=Counter(st/'count.json','test');counter.initialize()
        denied('unreserved_forward_denied',lambda:counter.compute('forward'))
        counter.start_role('diag');counter.reserve('diag');counter.compute('forward')
        denied('diag_optimizer_forbidden',lambda:counter.compute('optimizer'))
        denied('diag_backward_forbidden',lambda:counter.compute('backward'))
        denied('extra_forward_denied',lambda:counter.compute('forward'))
        counter.resolve()
        for _ in range(3):counter.reserve('diag');counter.compute('forward');counter.resolve()
        denied('diag_fifth_denied',lambda:counter.reserve('diag'));counter.finish_role()
        for role,n in [('cont',16),('split_a',8),('split_b',8)]:
            counter.start_role(role)
            for i in range(n):
                counter.reserve('update');counter.compute('forward');counter.compute('backward');counter.compute('optimizer');counter.resolve()
            denied(role+'_excess_denied',lambda:counter.reserve('update'));counter.finish_role()
        check('32_cumulative_limit',Counter(st/'count.json','test').read()['updates_reserved']==32)
        denied('restart_counter_denied',lambda:counter.initialize())
        denied('binding_mismatch_denied',lambda:Counter(st/'count.json','wrong').read())
        fail=Counter(st/'failure.json','test');fail.initialize();v=fail.read();v['completed_roles']=['diag'];atomic(fail.path,v);fail.start_role('cont');fail.reserve('update');fail.stop('simulated interruption')
        check('unknown_charged_not_refunded',fail.read()['updates_reserved']==fail.read()['unknown_updates']==1)
        denied('failed_restart_denied',lambda:fail.reserve('update'))
    import torch
    from candidate import seed_cpu,make_trainer,legacy_digest
    from fast_observer import digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    baseline=[json.loads(l) for l in (N0/'attempts.jsonl').read_text().splitlines()][:16]
    fixture_sources={str(fixture_path):sha(fixture_path)}
    for i,row in enumerate(manifest['batches'][:16]):
        check('fixture%d_sha'%(i+1),sha(row['path'])==row['sha256'])
        b=torch.load(row['path'],map_location='cpu',weights_only=False)
        got={'ids':list(map(str,b['keys'])),'input_sha256':digest(b['data']),'target_sha256':digest(b['target'])}
        check('fixture%d_native_pair'%(i+1),all(got[k]==row[k]==baseline[i][k] for k in got) and set(got['ids'])<=set(q['train_ids']))
        fixture_sources[row['path']]=row['sha256']
        if i==0:first=b
    seed_cpu();t=make_trainer('final',torch.device('cpu'),cpu/'native',BASE)
    check('actual_candidate_initializer',legacy_digest(t.network.state_dict())==read(N0/'initialization.json')['model'])
    before=legacy_digest(t.network.state_dict());t.network.eval()
    x=first['data'][...,256:768,512:1024].contiguous()
    with torch.no_grad():
        outputs=t.network(x);t.network.decoder.deep_supervision=False;single=t.network(x);t.network.decoder.deep_supervision=True
    check('final_inference_contract_exact',isinstance(single,torch.Tensor) and torch.equal(single,outputs[0]))
    check('CPU_no_parameter_updates',legacy_digest(t.network.state_dict())==before and not t.optimizer.state)
    check('CUDA_never_initialized',not torch.cuda.is_initialized())
    prior=read(STAGE/'cpu_attempt_02/CPU_REVIEW.json')
    check('prior86_all_pass',len(prior['checks'])==86 and all(prior['checks'].values()))
    for p,h in prior['source_hashes'].items():check('prior_source_'+Path(p).name,sha(p)==h)
    resource=resources('final');check('GPU0_idle',resource['used_MiB']<512)
    # Bind all actually imported project/native Python sources plus immutable configs and fixtures.
    runtime={}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and f.endswith('.py') and str(ROOT) in f and (str(CODE) in f or any(s in f for s in ['/nnunetv2/','/dynamic_network_architectures/'])):runtime[f]=sha(f)
    for p in BRANCH.glob('*.py'):runtime[str(p)]=sha(p)
    allsources={str(STAGE/'DRAFT_DESIGN.json'):sha(STAGE/'DRAFT_DESIGN.json'),str(STAGE/'AUTHORIZATION.json'):sha(STAGE/'AUTHORIZATION.json')}
    for p,h in prior['source_hashes'].items():
        if p not in fixture_sources:allsources[p]=h
    bd={'protocol_sha256':sha(STAGE/'ENGINEERING_PROTOCOL.json'),'runtime_sources':runtime,'all_sources':allsources,'fixture_sources':fixture_sources}
    bd['binding_sha256']=canonical(bd);atomic(STAGE/'source_binding.json',bd,new=True)
    atomic(cpu/'checks.json',{'passed':all(checks.values()),'checks':checks,'GPU_calls':0,'optimizer_updates':0,'CPU_forwards':2,'resource':resource},new=True)
    atomic(STAGE/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','binding_sha256':bd['binding_sha256'],'checks_sha256':sha(cpu/'checks.json'),'checks':len(checks),'prior86_reused':True},new=True)
    from engineering import Counter
    key=canonical({'binding':bd['binding_sha256'],'mode':'final','phase':'engineering'})
    Counter(out/'budget.json',key).initialize()
    update_authority(state='CPU_ENTRY_PASS_GPU_RELEASE_PENDING',binding_sha256=bd['binding_sha256'],CPU_review_sha256=sha(STAGE/'CPU_REVIEW.json'))
    print(json.dumps({'status':'CPU_ENTRY_PASS','checks':len(checks),'sources':len(runtime),'resource':resource,'GPU_calls':0}),flush=True)

if __name__=='__main__':main()
