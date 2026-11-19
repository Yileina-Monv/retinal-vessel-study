"""Same producer seeds as accepted run, with a real augmenter subclass for native cleanup."""
import random
from nnunet2d_scientific_runtime import set_all_seeds

def install_seeded_worker_adapter(seed,trainer_module=None,augmenter_module=None):
    if type(seed)is not int or seed not in [2026,2027,2028]:raise RuntimeError('unexpected seed adapter scope')
    if trainer_module is None:
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
        import batchgenerators.dataloading.nondet_multi_threaded_augmenter as augmenter_module
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        if int(get_allowed_n_proc_DA())!=1:raise RuntimeError('exactly one augmentation worker required')
    original=trainer_module.NonDetMultiThreadedAugmenter;producer=augmenter_module.producer;used=[]
    def seeded_producer(queue,data_loader,transform,thread_id,worker_seed,abort_event,pause_event=None,wait_time=.02):
        if worker_seed is not None:random.seed(int(worker_seed))
        return producer(queue,data_loader,transform,thread_id,worker_seed,abort_event,pause_event,wait_time)
    class SeededAugmenter(original):
        def __init__(self,*args,**kwargs):
            if len(used)>=2:raise RuntimeError('unexpected extra augmentation worker pool')
            n=int(kwargs.get('num_processes',args[2] if len(args)>2 else 1))
            if n!=1:raise RuntimeError('augmentation worker count changed')
            value=seed+len(used);kwargs['num_processes']=1;kwargs['seeds']=[value]
            used.append({'pool':'train' if len(used)==0 else 'internal_train_monitor','worker_count':1,'seeds':[value]})
            super().__init__(*args,**kwargs)
    augmenter_module.producer=seeded_producer;trainer_module.NonDetMultiThreadedAugmenter=SeededAugmenter
    return used
