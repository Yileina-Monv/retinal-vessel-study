"""One detached fixed2027 then2028 sequence; scientific outcomes never select the second seed."""
import os,sys,json,time,subprocess,secrets,signal,shutil,datetime
import replication_gate as g
from retinal_m0.maples_g1_safety import atomic_json,read_json,exclusive_lock,process_identity,owner_alive
from fast_observer import Journal,AmpLimits
from progress_watchdog import ProgressWatchdog,train_progress_token,inference_progress_token

def record_header(header,folder):
    backup=g.GROUP/folder;backup.mkdir(exist_ok=False)
    for p in [g.CODE/'STATUS.md',g.CODE/'项目全局状态与研究决策.md']:
        for _ in range(5):
            old=p.read_bytes();(backup/p.name).write_bytes(old);first,_,rest=old.decode().partition('\n');tmp=p.with_name(p.name+'.nnrep_record.'+str(os.getpid())+'.pending');tmp.write_text(first+'\n\n'+header+'\n\n'+rest.lstrip('\n'))
            if p.read_bytes()!=old:tmp.unlink();continue
            os.replace(tmp,p);break
        else:raise RuntimeError('concurrent management edits')
def wait_resources(seed):
    while True:
        try:return g.resources()
        except RuntimeError as e:
            if 'GPU0 busy' not in str(e):raise
            atomic_json(g.GROUP/'job_state.json',{'state':'WAITING_GPU0','seed':seed,'time':time.time(),'reason':str(e),'no_eviction_or_GPU_switch':True});time.sleep(30)
def release_seed(seed):
    g.set_context(seed);key=g.binding_key();j=Journal(g.STAGE/'budget.json',key,100,250,50,AmpLimits(5,25));s=j.read()
    if s['attempts'] or s['monitor_attempts'] or s['status']!='READY' or (g.STAGE/'TRAIN_SESSION.json').exists():raise RuntimeError('seed already attempted; no reset or auto resume')
    if seed==2028 and not (g.GROUP/'runs/seed_2027/completion.json').exists():raise RuntimeError('2027 must technically complete first; scientific effect never cancels2028')
    r=g.expected_release();atomic_json(g.STAGE/'GPU_RELEASE.json',r);v=g.authority();runs=v['runs'];runs[str(seed)].update(state='RELEASED',release_sha256=g.sha(g.STAGE/'GPU_RELEASE.json'),binding_sha256=key);g.update_group(state='SERIAL_RUNNING',active_seed=seed,GPU_released=True,runs=runs,next='Run fixed2027 then2028 onGPU0; scientific outcome cannot cancel the next seed, technical failure pauses; no extra experiments/Test')
    g.validate(full=True);return key
def run_child(role,key):
    stage=g.STAGE;seed=g.SEED;session=stage/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json')
    if session.exists():raise RuntimeError('role already attempted; no auto retry')
    nonce=secrets.token_hex(24);env=dict(os.environ,RETINA_NN_SEED=str(seed),CUDA_VISIBLE_DEVICES=g.GPU,PYTHONHASHSEED=str(seed),NNUNET100_NONCE=nonce,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',nnUNet_compile='false',nnUNet_n_proc_DA='1',MPLCONFIGDIR=str(stage/'matplotlib'),nnUNet_raw=str(g.DATA_BASE/'raw'),nnUNet_preprocessed=str(g.DATA_BASE/'preprocessed'),nnUNet_results=str(stage/'nnunet_results'))
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    script=g.BRANCH/('replication_train.py' if role=='train' else 'replication_infer.py');flag='--authorized-train' if role=='train' else '--authorized-final-inference';begin=time.monotonic()
    with (stage/(role+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-u','-B',str(script),flag,'--parent-pid',str(os.getpid()),'--nonce',nonce],cwd=g.CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owner=process_identity(p.pid);atomic_json(session,{'controller':process_identity(),'worker':owner,'nonce':nonce,'binding_sha256':key,'role':role,'seed':seed})
        g.update_status(execution_started=True,worker_pid=p.pid,controller_pid=os.getpid(),active_role=role)
        watchdog=ProgressWatchdog(1800);last_warning=-float('inf')
        try:
            while p.poll() is None:
                try:
                    pbudget=stage/('budget.json' if role=='train' else 'inference_budget.json');observed={'waiting_for_budget':True}
                    if pbudget.exists():
                        counters=read_json(pbudget);token=train_progress_token(counters) if role=='train' else inference_progress_token(counters);observed=watchdog.observe(token,time.monotonic())
                        if observed['stop'] and time.monotonic()-last_warning>=300:print('PROGRESS_WARNING_ONLY',seed,role,observed,flush=True);last_warning=time.monotonic()
                    atomic_json(g.GROUP/'job_state.json',{'state':'RUNNING','seed':seed,'role':role,'worker':owner,'elapsed_role_seconds':time.monotonic()-begin,'time':time.time(),'normal_progress_wall_limit':None,'heartbeat_alert_only':True,'watchdog':observed})
                except Exception as e:print('MONITOR_IO_WARNING_ONLY',repr(e),flush=True)
                time.sleep(5)
        except BaseException:
            if owner_alive(owner):
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    result={'seed':seed,'role':role,'exit_code':p.returncode,'elapsed_process_seconds':time.monotonic()-begin};atomic_json(stage/(role+'_process_result.json'),result)
    if p.returncode:raise RuntimeError('seed'+str(seed)+' '+role+' worker stopped '+str(p.returncode))
    return result
def aggregate():
    import numpy as np
    ds={s:read_json(g.GROUP/f'runs/seed_{s}/analysis/endpoint_100epochs.json') for s in [2027,2028]};keys=list(ds[2027]['metrics']);summary={}
    for m in keys:
        values=[ds[s]['metrics'][m]['NNUNET-A0']['estimate'] for s in [2027,2028]];summary[m]={'per_seed_delta':dict(zip(['2027','2028'],values)),'mean_delta':float(np.mean(values)),'sample_SD_ddof1_n2':float(np.std(values,ddof=1))}
    dice_ok=all(d['metrics']['dice']['NNUNET-A0']['lower']>0 and d['metrics']['dice']['NNUNET-A0']['estimate']>0 for d in ds.values());guards_ok=all(d['readout']['mean_guards_nonadverse'] and d['readout']['strict_guard_intervals_pass'] for d in ds.values())
    out={'status':'COMPLETE_PENDING_SOL_REVIEW','seeds':[2027,2028],'metrics':summary,'Dice_replication_signal_both_seeds':dice_ok,'protected_Dice_signal_both_seeds':dice_ok and guards_ok,'scope':'prospective two-seed repeated-Dev workflow replication, no independent Test; old2026 interrupted queue trajectory context only','two_seeds_not56_independent_images':True,'Test_accessed':False,'automatic_extra_experiment':False}
    context={}
    for label,path in [('nnUNet2026',g.group_protocol()['original_2026_result']),('SCSE2026',str(g.ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scse_single_module_2000_v1_20261026/runs/seed_2026/SCSE/analysis/endpoint_2000.json'))]:
        cp=__import__('pathlib').Path(path);old=read_json(cp);proof=read_json(cp.parent/'SOL_POSTRUN_RESEARCH_REVIEW.json')
        if g.sha(cp)!=proof['endpoint_sha256']:raise RuntimeError('historical context result changed')
        method='NNUNET' if label=='nnUNet2026' else 'SCSE';context[label]={'Dice':old['absolute_means'][method]['dice'],'A0_Dice':old['absolute_means']['A0']['dice'],'source':path,'sha256':g.sha(cp),'separate_exploratory_context_only':True}
    out['historical_context']=context;out['run_costs']={str(s):read_json(g.GROUP/f'runs/seed_{s}/train_process_result.json') for s in [2027,2028]}
    atomic_json(g.GROUP/'aggregate.json',out);lines=['# nnU-Net固定2027/2028复现：待Sol科研复盘','','|seed|A0 Dice|nnU-Net Dice|差值pp|95%区间pp|严格保护|','|---|---:|---:|---:|---|---|']
    for seed,d in ds.items():
        x=d['metrics']['dice']['NNUNET-A0'];lines.append(f'|{seed}|{100*d["absolute_means"]["A0"]["dice"]:.4f}|{100*d["absolute_means"]["NNUNET"]["dice"]:.4f}|{100*x["estimate"]:+.4f}|[{100*x["lower"]:+.4f},{100*x["upper"]:+.4f}]|{d["readout"]["strict_guard_intervals_pass"]}|')
    lines+=['','Dice复现信号：'+str(dice_ok)+'；受保护信号：'+str(dice_ok and guards_ok)+'。均值方向与区间跨零分别解释。','','历史探索参照（单独列出，不参与两新seed主判读）：','|模型seed2026|Dice%|原A0 Dice%|','|---|---:|---:|']
    for label,v in context.items():lines.append(f'|{label}|{100*v["Dice"]:.4f}|{100*v["A0_Dice"]:.4f}|')
    lines+=['','旧2026及scSE仅参照，不是同执行协议三seed或单变量模型消融；官方Test封存，不自动扩实验。训练进程时长与显存逐seed报告，A0使用M0预训练的费用不混同为只有微调费用。'];(g.GROUP/'report.md').write_text('\n'.join(lines)+'\n');return out
def main():
    g.group_protocol();r=read_json(g.GROUP/'CPU_REVIEW.json');b=read_json(g.GROUP/'source_binding.json');authorization=read_json(g.GROUP/'SERIAL_RELEASE.json')
    if r['status']!='CPU_ACCEPTED' or authorization!={'seeds':[2027,2028],'group_protocol_sha256':g.GROUP_PROTOCOL_SHA,'group_binding_sha256':b['binding_sha256'],'CPU_review_sha256':g.sha(g.GROUP/'CPU_REVIEW.json'),'GPU':g.GPU,'total_attempt_cap':50000,'order':[2027,2028],'no_outcome_selection':True,'no_automatic_retry':True,'Test_allowed':False}:raise RuntimeError('serial independent release missing/stale')
    with exclusive_lock(g.GROUP/'SERIAL_RUNTIME.lock'):
        if (g.GROUP/'authorization_consumed.json').exists():raise RuntimeError('serial group already attempted')
        atomic_json(g.GROUP/'authorization_consumed.json',{'controller':process_identity(),'release_sha256':g.sha(g.GROUP/'SERIAL_RELEASE.json'),'time':time.time()})
        for seed in [2027,2028]:
            try:
                with exclusive_lock(g.ROOT/'experiments/a14_gpu0.lock'):
                    wait_resources(seed);key=release_seed(seed);train=run_child('train',key);g.validate(full=True)
                    completion=read_json(g.STAGE/'training_completion.json');assert completion['attempts']==25000 and completion['internal_train_monitor_batches']==5000
                    wait_resources(seed);infer=run_child('infer',key)
                g.update_group(GPU_released=False);g.update_status(state='GPU_COMPLETE',worker_pid=None,active_role='CPU_ANALYSIS')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',RETINA_NN_SEED=str(seed));p=subprocess.run([sys.executable,'-u','-B',str(g.BRANCH/'analysis_runner.py'),'--seed',str(seed)],cwd=g.CODE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                (g.STAGE/'analysis.log').write_text(p.stdout)
                if p.returncode:raise RuntimeError('CPU analysis failed; no inference replay')
                endpoint=read_json(g.STAGE/'analysis/endpoint_100epochs.json');atomic_json(g.STAGE/'completion.json',{'status':'COMPLETE_PENDING_SOL_REVIEW','seed':seed,'training':train,'inference':infer,'readout':endpoint['readout'],'endpoint_path':str(g.STAGE/'analysis/endpoint_100epochs.json'),'Test_accessed':False});g.update_status(state='CONSUMED',training_complete=True,active_role=None,run_state='COMPLETE_PENDING_SOL_REVIEW',result_path=str(g.STAGE/'analysis/endpoint_100epochs.json'))
                record_header(f'### nnU-Net复现seed{seed}训练及终点评价完成，按预定排程继续\n\n固定100epochs/25000尝试与Train监控完成，唯一final/.5 Dev28及CPU报告已生成。Dice {100*endpoint["absolute_means"]["NNUNET"]["dice"]:.4f}% vs对应A0 {100*endpoint["absolute_means"]["A0"]["dice"]:.4f}%，原保护另报，不按效果选择下一seed。自动事实归档待Sol科研复盘；官方Test封存。报告 '+str(g.STAGE/'analysis/report.md'),'management_before_seed'+str(seed)+'_completion')
            except BaseException as e:
                atomic_json(g.GROUP/'failure.json',{'seed':seed,'error':repr(e),'time':time.time(),'no_automatic_retry':True});g.update_status(state='FAILED_STOP',worker_pid=None,run_state='FAILED_STOP');g.update_group(state='FAILED_STOP',GPU_released=False,next='Technical failure pauses fixed sequence; preserve all counters, no retry/reset or extra budget. Sol review required; scientific outcome alone never cancels2028.');raise
        summary=aggregate();g.update_group(state='CONSUMED',active_seed=None,GPU_released=False,next='Both fixed seeds complete; Sol reviews paired Dice/protection and costs. No auto seed/module/threshold/steps/Test.',scientific_review_pending=True,result_path=str(g.GROUP/'aggregate.json'));atomic_json(g.GROUP/'job_state.json',{'state':'COMPLETE_PENDING_SOL_REVIEW','time':time.time()});record_header('### nnU-Net固定2027/2028双seed复现全部完成，待Sol科研复盘\n\n两条均按预定顺序完成，未按第一条效果取消第二条。对照各自A0，不重复A0推理；逐seed指标/图像配对区间、两seed均值与样本SD(n=2)已归档，旧2026与scSE单列上下文，不作56独立图片或严格同协议三seed独立验证。官方Test封存，不自动追加实验。汇总 '+str(g.GROUP/'report.md'),'management_before_group_completion')

if __name__=='__main__':
    p=__import__('argparse').ArgumentParser();p.add_argument('--run-authorized-serial',required=True,action='store_true');p.parse_args();main()
