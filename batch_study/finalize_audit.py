"""Validate completed artifacts and export transparent per-image observations."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch
from retinal_batchstudy.run import CONFIG,read,dump,binding
from retinal_m0.state import state_digest

ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'runs/batch_study_v1/rtx5080'
config=read(CONFIG)
fingerprints=binding()
assert read(root/'binding.json')==fingerprints
assert read(root/'data_check.json')['group_overlap']==0
records=[]
audits=[]
total_update_seconds=total_validation_seconds=0.
for seed in config['seeds']:
    sequence_digests=set()
    initial_digests=set()
    for arm in config['arms']:
        directory=root/f'seed_{seed}'/arm['id']
        result=read(directory/'result.json')
        saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
        assert saved['binding']==fingerprints
        assert saved['config']==config and saved['arm']==arm and saved['seed']==seed
        assert saved['consumed']==result['draws']==config['training_draws']
        assert saved['step']==result['updates']==config['training_draws']//(arm['microbatch']*arm['accumulation'])
        assert len(saved['signatures'])==config['training_draws']
        assert [r['step'] for r in saved['history']]==list(range(1,saved['step']+1))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in saved['history'])
        assert result['amp_skipped_updates']==0
        signature=hashlib.sha256(json.dumps(saved['signatures'],sort_keys=True).encode()).hexdigest()
        assert signature==result['sample_sequence_digest']
        sequence_digests.add(signature)
        initial_digests.add(result['initial_digest'])
        evaluation=read(directory/f"eval_draw_{config['training_draws']}.json")
        assert evaluation==result['final']
        assert len(evaluation['rows'])==8 and len({r['key'] for r in evaluation['rows']})==8
        for row in evaluation['rows']:
            assert abs(row['dice']-2*row['tp']/(2*row['tp']+row['fp']+row['fn']))<1e-12
            records.append({'seed':seed,'arm':arm['id'],**{k:row[k] for k in ('key','disease','dice','precision','recall','tp','fp','fn')}})
        assert abs(evaluation['macro_dice']-statistics.mean(r['dice'] for r in evaluation['rows']))<1e-12
        initial=torch.load(root/f'initial_seed_{seed}.pt',map_location='cpu',weights_only=True)
        assert state_digest(initial['model'])==result['initial_digest']
        total_update_seconds+=result['updates_seconds']
        total_validation_seconds+=result['validation_seconds']
        audits.append({'seed':seed,'arm':arm['id'],'updates':saved['step'],'checkpoint_and_result_consistent':True})
        del saved,initial
    assert len(sequence_digests)==len(initial_digests)==1
out=ROOT/'batch_study/results/rtx5080'
with (out/'per_image_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
audit={'status':'passed','completed_runs':len(audits),'per_image_records':len(records),'test_external_images_opened_by_this_study':0,
       'training_crops_total':len(audits)*config['training_draws'],'updates_total':sum(r['updates'] for r in audits),
       'training_loop_seconds_total':total_update_seconds,'validation_seconds_accounted_total':total_validation_seconds,
       'timing_note':'Validation accounting includes an initial evaluation allowance for each arm; cached/shared initial evaluations are not all rerun. This is not end-to-end elapsed wall time.',
       'same_seed_initial_weights_and_actual_input_sequences_match':True,'finite_losses_and_gradients':True,'amp_skipped_updates':0,'runs':audits}
dump(out/'artifact_audit.json',audit)
print({k:v for k,v in audit.items() if k!='runs'})
