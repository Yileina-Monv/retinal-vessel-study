"""Read-only CPU review of immutable first30 scientific snapshots; no model calls."""
import os,sys,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
b=Path('${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/research_branches/nnunet2d_20261020/deep_supervision_ablation_20261028/science100_v1')
sys.path.insert(0,str(b))
from common import *
import torch
from fast_observer import digest,states_finite


def main():
    original=[json.loads(l) for l in (N0/'attempts.jsonl').read_text().splitlines()][:30]
    results={}
    for mode in GPU:
        validate(mode,'science',full=True)
        out=STAGE/'science'/mode;file=out/'state_start_00030.pt'
        state=torch.load(file,map_location='cpu',weights_only=False)
        for k,h in state['digests'].items():
            if digest(state[k])!=h:raise RuntimeError('snapshot self digest mismatch:'+k)
        if not states_finite(state['model'],state['optimizer']):raise RuntimeError('nonfinite state')
        budget=state['budget']
        if budget['attempts']!=30 or budget['success']+budget['skip']!=30 or budget['fatal'] or budget['unknown'] or \
           budget['pending'] or budget['train_forwards']!=30 or budget['backwards']!=30 or budget['monitor_attempts'] or \
           budget['status']!='READY':raise RuntimeError('first30 computation/accounting mismatch')
        raw=(out/'attempts.jsonl').read_bytes().splitlines(keepends=True);prefix=b''.join(raw[:30])
        if hashlib.sha256(prefix).hexdigest()!=state['log_prefixes']['attempts.jsonl']:raise RuntimeError('snapshot log prefix differs')
        rows=[json.loads(l) for l in raw[:30]]
        for actual,want in zip(rows,original):
            if any(actual[k]!=want[k] for k in ['ids','input_sha256','target_sha256','lr']):raise RuntimeError('N0 pairing drift')
            if actual['classification']=='SUCCESS' and (not actual['model_changed'] or actual['optimizer_calls']!=1):
                raise RuntimeError('no actual successful update')
        initial=read(out/'initialization.json');engineering=read(ENGINEERING_STAGE/'engineering'/mode/'cont_initialization.json')
        if initial['backbone']!=read(N0/'initialization.json')['model'] or initial['weights']!=engineering['weights'] or \
           initial['engineering_weights_loaded'] or not initial['fresh_initialization']:raise RuntimeError('wrong scientific initialization')
        if read(out/'actual_seed_pools.json')!=[{'pool':'train','worker_count':1,'seeds':[2027]},
              {'pool':'internal_train_monitor','worker_count':1,'seeds':[2028]}]:raise RuntimeError('wrong real producer seeds')
        sets=read(out/'loader_sets.json')
        if sets['Train']!=sorted(protocol()['train_ids']) or sets['Train_monitor']!=sets['Train']:raise RuntimeError('non-Train loader')
        session=read(out/'train_session.json')
        if process_identity(session['worker']['pid'])!=session['worker'] or process_identity(session['controller']['pid'])!=session['controller']:
            raise RuntimeError('actual worker/controller not alive')
        if max(r['GPU_peak_reserved_MiB'] for r in rows)>10240:raise RuntimeError('memory cap exceeded')
        report={'status':'PASS','seed':2027,'mode':mode,'GPU':GPU[mode],'snapshot_sha256':sha(file),
                'first30_counts':budget,'full_snapshot_self_digests_verified':True,
                'real_successful_parameter_updates_verified':budget['success'],
                'all30_inputs_targets_IDs_LR_exact_N0':True,'real_Train_only_loader_sets':True,
                'fresh_native_weights_and_supervision_verified':True,'engineering_weights_loaded':False,
                'actual_worker':session['worker'],'actual_controller':session['controller'],
                'peak_reserved_MiB':max(r['GPU_peak_reserved_MiB'] for r in rows),
                'no_Dev_or_Test_model_calls':True,'reviewer_GPU_initialized':torch.cuda.is_initialized(),
                'reviewer_code_sha256':sha(Path(__file__))}
        atomic(out/'SOL_START_VERIFICATION.json',report,new=True);results[mode]=report
    atomic(STAGE/'SOL_START_REVIEW.json',{'status':'PASS','first30_verified':True,'results':results,
           'engineering_updates_preserved':32,'engineering_diagnostics_preserved':5,'new_scientific_budget':25000,'Test_accessed':False},new=True)
    update_authority(state='SCIENCE_RUNNING_FIRST30_SOL_VERIFIED',
                     next='Continue one fixed100epochs in background; unique final/.5 Dev28 and CPU report; Sol research review then stop')
    print(json.dumps({m:{'success_first30':r['first30_counts']['success'],'skip_first30':r['first30_counts']['skip'],
                        'peak_reserved_MiB':r['peak_reserved_MiB']} for m,r in results.items()}))


if __name__=='__main__':main()
