"""Complete the interrupted epoch without repeating its LR step or discarding its outputs."""
import torch
def run_remainder(trainer,observer,journal,first_batch):
    initial=journal.read()
    if initial["phase"]!="train" or initial["pending"] is not None or trainer.current_epoch!=initial["epoch"]:
        raise RuntimeError("unsupported recovery cursor")
    first=True
    while trainer.current_epoch<trainer.num_epochs:
        s=journal.read()
        if s["iteration"]==0:
            trainer.on_epoch_start()
            trainer.on_train_epoch_start()
        else:
            if not first:raise RuntimeError("unexpected partial epoch")
            if len(observer.partial_train)!=s["iteration"]:raise RuntimeError("missing saved partial outputs")
            trainer.network.train()
        outputs=list(observer.partial_train)
        while len(outputs)<trainer.num_iterations_per_epoch:
            batch=first_batch if first else next(trainer.dataloader_train)
            first=False
            outputs.append(trainer.train_step(batch))
        trainer.on_train_epoch_end(outputs)
        with torch.no_grad():
            trainer.on_validation_epoch_start()
            val_outputs=[]
            for _ in range(trainer.num_val_iterations_per_epoch):
                val_outputs.append(trainer.validation_step(next(trainer.dataloader_val)))
            trainer.on_validation_epoch_end(val_outputs)
        trainer.on_epoch_end()
    trainer.on_train_end()
