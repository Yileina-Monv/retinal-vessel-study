"""CPU-only admission, persistent ledger and real toy-parameter gradient tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import common as c, tempfile,copy,types,ast
from unittest.mock import patch
import torch
from fast_observer import Journal,AmpLimits,StepEvidence,digest,rng_pack
from nl_observer import NLObserver
from aux_loss import sparse_aux,total_loss
from controller import require_unused
from infer import verify_tile_count,prevent_native_retry
results=[]
def ok(v,name):
 if not v:raise AssertionError(name)
 results.append(name)
def reject(fn,name):
 try:fn()
 except (RuntimeError,ValueError,KeyError,FileNotFoundError):results.append(name);return
 raise AssertionError('not rejected: '+name)
original_group=c.GROUP;original_code=c.CODE;q=card()
validate_document(q);ok(True,'approved scope accepted')
for path,value in [(('authorization','science_training_approved'),False),(('authorization','Test_allowed'),True),(('training','seeds'),[2027]),(('training','train_attempts_per_seed'),25001),(('method','lambda'),.03),(('method','max_positive_groups_per_crop'),16),(('runtime','device'),'GPU1'),(('gradients','ratio_stop_strictly_greater_than'),3),(('runtime','train_hours_cap_per_seed'),51)]:
 bad=copy.deepcopy(q);bad[path[0]][path[1]]=value;reject(lambda:validate_document(bad),'scope refusal '+str(path))
with tempfile.TemporaryDirectory(prefix='nl-science-cpu-') as d:
 d=Path(d);c.GROUP=d;c.CODE=d
 (d/'STATUS.md').write_text(c.MARKER);atomic(d/'protocol.json',q);atomic(d/'CPU_REVIEW.json',{'status':'PASS'})
 source=d/'source.py';source.write_text('x=1\n');atomic(d/'SOURCE_BINDING.json',{'runtime_sources':{str(source):sha(source)},'immutable_sources':{}})
 release={'scope':'NL_TWO_SEED_SCIENCE_ONLY','protocol_sha256':sha(d/'protocol.json'),'binding_sha256':sha(d/'SOURCE_BINDING.json'),'cpu_review_sha256':sha(d/'CPU_REVIEW.json')};atomic(d/'SCIENCE_RELEASE.json',release)
 ok(validate(True)==sha(d/'SCIENCE_RELEASE.json'),'bound release accepted')
 for key in ['scope','protocol_sha256','binding_sha256','cpu_review_sha256']:
  bad=dict(release);bad[key]='INVALID';atomic(d/'SCIENCE_RELEASE.json',bad);reject(lambda:validate(True),'release refuses '+key)
 atomic(d/'SCIENCE_RELEASE.json',release)
 source.write_text('x=2\n');reject(lambda:validate(True),'runtime code drift rejected');source.write_text('x=1\n')
 (d/'STATUS.md').write_text('HOLD');reject(lambda:validate(True),'missing STATUS release rejected');(d/'STATUS.md').write_text(c.MARKER)
 import controller
 old=controller.GROUP;controller.GROUP=d
 require_unused();atomic(d/'SESSION.json',{});reject(require_unused,'one shot launch rejects consumed session');controller.GROUP=old
 c.GROUP=original_group;c.CODE=original_code;c._cache['time']=0
 # Persistent accounting, no model calls.
 j=Journal(d/'ledger.json','test',1,1,1,AmpLimits(5,25));j.initialize()
 reject(j.initialize,'ledger reset refused');reject(lambda:j.compute('optimizer'),'optimizer before reservation refused')
 j.start('train',0);ok(j.read()['attempts']==1,'attempt persisted before compute');reject(lambda:j.compute('optimizer'),'optimizer before backward refused')
 j.compute('forward');reject(lambda:j.compute('forward'),'duplicate forward refused');j.compute('backward');reject(lambda:j.compute('backward'),'duplicate total backward refused');j.compute('optimizer');reject(lambda:j.compute('optimizer'),'duplicate optimizer refused')
 j.resolve_train(StepEvidence(1.,1.,128.,128.,1,True,True),True);j.finish_train();j.start('monitor',0);j.compute('forward');reject(lambda:j.compute('backward'),'monitor backward prohibited');reject(lambda:j.compute('optimizer'),'monitor update prohibited');j.resolve_monitor();j.finish_monitor();j.finish_epoch(1);reject(lambda:j.start('train',1),'completed budget cannot extend')
 k=Journal(d/'unknown.json','test',1,2,1,AmpLimits(5,25));k.initialize();k.start('train',0);k.compute('forward');k.stop('CPU simulated interruption');ok(k.read()['attempts']==1 and k.read()['unknown']==1,'failure remains charged');reject(lambda:k.start('train',0),'failed trajectory cannot continue')
 for label,limits,interleave in [('consecutive',AmpLimits(5,25),False),('cumulative',AmpLimits(5,25),True)]:
  k=Journal(d/(label+'.json'),'test',1,60,1,limits);k.initialize();expected=25 if interleave else 5
  for n in range(expected):
   k.start('train',0);k.compute('forward');k.compute('backward');k.resolve_train(StepEvidence(1.,float('inf'),128.,64.,0,True,True),False)
   if interleave and n<expected-1:
    k.start('train',0);k.compute('forward');k.compute('backward');k.compute('optimizer');k.resolve_train(StepEvidence(1.,1.,64.,64.,1,True,True),True)
  ok(k.read()['status']=='STOPPED' and k.read()['skip']==expected,'AMP '+label+' enforced without clearing')
 # Actual CPU parameters, component autograd hooks and total backward accounting.
 net=torch.nn.Conv2d(1,2,1,bias=True);opt=torch.optim.SGD(net.parameters(),lr=.01)
 j=Journal(d/'gradient.json','test',1,1,1,AmpLimits(5,25));j.initialize();j.start('train',0)
 o=NLObserver.__new__(NLObserver);o.t=types.SimpleNamespace(network=net,optimizer=opt,grad_scaler=types.SimpleNamespace(get_scale=lambda:128.));o.j=j;o.authorize=lambda:None;o.diag_mode=False;o.active=1;o.force=None
 handles=[net.register_forward_pre_hook(o.forward),next(net.parameters()).register_hook(o.backward),opt.register_step_pre_hook(o.optimizer)]
 x=torch.ones(2,1,3,3);y=net(x);base=y.square().mean();groups=[[{'positive':[1,1],'negatives':[[0,0],[0,1]]}],[]];aux=sparse_aux(y,groups)
 before=digest(rng_pack());gb=o.diagnostic_grad(base,'base');ga=o.diagnostic_grad(aux,'aux')
 ok(digest(rng_pack())==before,'component diagnostics preserve RNG');ok(j.read()['backwards']==0 and j.read()['optimizer_calls']==0,'component hooks do not charge total backward/update');ok(all(p.grad is None for p in net.parameters()),'diagnostic does not accumulate grad')
 total=base+q['method']['lambda']*aux;total.backward();ok(j.read()['backwards']==1,'one total backward charged')
 expected=[a+q['method']['lambda']*b for a,b in zip(gb,ga)];ok(all(torch.allclose(p.grad,g,rtol=1e-5,atol=1e-6) for p,g in zip(net.parameters(),expected)),'real parameter gradient equals component sum')
 opt.step();ok(j.read()['optimizer_calls']==1,'only total update counted')
 for h in handles:h.remove()
 z=torch.zeros(2,2,3,3,requires_grad=True);empty=sparse_aux(z,[[],[]]);empty.backward();ok(float(empty)==0 and torch.count_nonzero(z.grad)==0,'empty full batch graph connected zero')
 z=torch.zeros(2,2,3,3,requires_grad=True);loss=sparse_aux(z,groups);loss.backward();ok(abs(float(loss)-.5*__import__('math').log(2))<1e-7,'empty crop retained in denominator');ok(z.grad[0,1,1,1]<0 and z.grad[0,1,0,0]>0,'positive negative gradient directions correct')
 base=torch.tensor(3.,requires_grad=True);ok(total_loss(base,z,groups,0) is base,'zero coefficient exact base identity')
for n,cap in [(1,3),(3,3)]:verify_tile_count(n,cap);ok(True,'valid inference tile bound')
reject(lambda:verify_tile_count(4,3),'inference overbudget refused')
state={'running':True,'calls':0}
def original():state['calls']+=1;raise RuntimeError('simulated native inference error')
p=types.SimpleNamespace(_internal_predict_sliding_window_return_logits=original)
prevent_native_retry(p,lambda e:state.update(running=False),lambda:state['running'])
reject(p._internal_predict_sliding_window_return_logits,'first inference failure retained');reject(p._internal_predict_sliding_window_return_logits,'native inference fallback refused');ok(state['calls']==1,'no duplicate inference call')
for f in BRANCH.glob('*.py'):ast.parse(f.read_text());ok(True,'syntax '+f.name)
ok(read(GROUP/'cpu_integration/RESULT.json')['status']=='PASS','native64 batch integration PASS')
ok(not torch.cuda.is_initialized(),'no CUDA initialized')
result={'status':'PASS','checks':len(results),'passed':results,'GPU_calls':0,'scientific_parameter_updates':0,'synthetic_CPU_optimizer_steps':1,'Dev_Test_read':False,'limitations':['CPU toy gradients are not a new full-network GPU acceptance','Prior accepted128-update GPU engineering reused; native64batch CPU stream verified','Native asynchronous queue exact resume unsupported']}
atomic(GROUP/'CPU_ENTRY_TESTS.json',result,True);print(json.dumps(result,ensure_ascii=False),flush=True)
