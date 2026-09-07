import copy
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
import uuid
from unittest.mock import patch
import numpy as np
import torch
from retinal_release.common import ROOT,identity,write,safe,digest
from retinal_release.freeze import make_fov,split
from retinal_release.data import masked_loss
from retinal_release import exchange

@contextmanager
def fixture_directory():
    # Keep fixtures for audit; Python 3.13's Windows tempfile mode 0700 is
    # incompatible with this desktop sandbox token. Normal inherited ACLs work.
    path=ROOT/'.runtime/release_test_fixtures'/uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path

class ReleaseTests(unittest.TestCase):
    def test_masked_accumulation_unequal_fov_and_empty(self):
        torch.manual_seed(7)
        x=torch.randn(4,1,8,8,requires_grad=True);target=(torch.rand_like(x)>.8).float()
        fov=torch.ones_like(x);fov[0]=0;fov[1,:,:4]=0;fov[2,:,:7]=0
        masked_loss(x,target,fov).backward();expected=x.grad.clone();x.grad=None
        for i in range(4):(masked_loss(x[i:i+1],target[i:i+1],fov[i:i+1])/4).backward()
        torch.testing.assert_close(x.grad,expected,rtol=1e-5,atol=1e-7)
        self.assertEqual(float(x.grad[0].abs().sum()),0)
    def test_dark_sector_is_inside_fov(self):
        y,x=np.ogrid[:128,:128];circle=(y-63.5)**2+(x-63.5)**2<50**2
        image=np.zeros((128,128,3),np.uint8);image[circle]=100
        image[64:,64:]=0
        result=make_fov(image)
        self.assertTrue(result[85,85]);self.assertFalse(result[0,0])
    def test_previous_training_cannot_enter_validation(self):
        rows=[dict(key=str(i),disease='A',group_size=1,label_is_empty=0,trial_split='smoke_train' if i<3 else 'smoke_val' if i==3 else 'deferred') for i in range(20)]
        result=split(rows,dict(validation_count=4,split_seed=123))
        self.assertTrue(all(r['trial_split']=='train' for r in result[:3]))
        self.assertEqual(result[3]['trial_split'],'validation')
    def test_path_traversal_rejected(self):
        for p in ('../a','C:/a','a\\b','/a','a/../b'):
            with self.assertRaises(ValueError):safe(Path.cwd(),p)
    def test_registry_rejects_stale_conflicting_and_copied_authority(self):
        with fixture_directory() as tmp:
            root=Path(tmp);state=root/'authority';run=root/'run';run.mkdir()
            write(root/'releases/development_v1/release.json',dict(release_id='fixture'))
            with patch.object(exchange,'ROOT',root):ticket=exchange.issue('job1',exchange.machine(),2026,2,'gpu',state=state)
            write(run/'ticket.json',ticket)
            def make(step,name):
                cp=dict(ticket_id=ticket['ticket_id'],release_id='fixture',machine=exchange.machine(),gpu_uuid='gpu',
                    step=step,consumed=step*4,model={},optimizer={},scaler={},rng={},history=[{}]*step,signatures=[{}]*(step*4))
                torch.save(cp,run/'last.pt');write(run/'progress.json',dict(step=step,status='completed' if step==2 else 'paused',checkpoint_sha256=digest(run/'last.pt')))
                exchange.pack(run,root/name);return root/name
            old=make(1,'old.zip');new=make(2,'new.zip')
            self.assertEqual(exchange.ingest(new,state=state)['status'],'accepted')
            self.assertEqual(exchange.ingest(new,state=state)['status'],'already_accepted')
            with self.assertRaises(ValueError):exchange.ingest(old,state=state)
            conflict=make(2,'conflict.zip')
            # Alter payload so a same-step divergent snapshot has a different content identity.
            cp=torch.load(run/'last.pt',weights_only=True);cp['model']={'different':torch.tensor(1)};torch.save(cp,run/'last.pt')
            write(run/'progress.json',dict(step=2,status='completed',checkpoint_sha256=digest(run/'last.pt')))
            exchange.pack(run,root/'different.zip')
            with self.assertRaises(ValueError):exchange.ingest(root/'different.zip',state=state)
            import shutil
            shutil.copytree(state,root/'copy')
            with self.assertRaises(ValueError):
                with exchange.authority(root/'copy'):pass
    def test_running_export_rejected(self):
        with fixture_directory() as tmp:
            root=Path(tmp);(root/'running.lock').touch()
            with self.assertRaises(FileExistsError):exchange.pack(root,root/'x.zip')

if __name__=='__main__':unittest.main()
