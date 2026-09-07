import unittest
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from retinal_c1.geometry import build, chains, select, coordinates
from retinal_c1.objectives import auxiliary,scnp_logits
from retinal_data.fives import paired_crop_transform


class CoreTests(unittest.TestCase):
    def fixture(self):
        s=np.zeros((144,144),bool);s[40,12:132]=True;s[92,12:132]=True
        label=binary_dilation(s,structure=disk(2));fov=np.ones_like(s)
        return s,label,fov

    def test_straight_parallel_ownership_and_nonoverlap(self):
        s,label,fov=self.fixture();geo,stats=build(label,fov,s,s)
        self.assertEqual(stats['chain_count'],2)
        pairs=select(geo,[0,0,144,144,0,0,0],7,limit=None)
        self.assertGreaterEqual(len(pairs),4)
        allpos=np.concatenate([p['pos'] for p in pairs])
        self.assertEqual(len(np.unique(allpos,axis=0)),len(allpos))
        for p in pairs:
            self.assertTrue(label[tuple(p['pos'].T)].all());self.assertFalse(label[tuple(p['bg'].T)].any())
            self.assertLess(np.max(abs(p['bg'][:,0]-p['pos'][0,0])),15)

    def test_corner_diagonals_do_not_create_junction(self):
        s=np.zeros((80,80),bool);s[20,10:51]=True;s[20:61,50]=True
        paths,stats=chains(s)
        self.assertEqual(stats['junction_pixels'],0);self.assertEqual(len(paths),1)

    def test_all_sparse_augmentations_and_crop_margin(self):
        s,label,fov=self.fixture();geo,_=build(label,fov,s,s)
        rgb=np.repeat(label[:,:,None].astype(np.uint8),3,2)
        for hf in (0,1):
            for vf in (0,1):
                for k in range(4):
                    g=[8,8,128,128,hf,vf,k]
                    _,mask=paired_crop_transform(rgb,label,top=8,left=8,size=128,hflip=hf,vflip=vf,quarter_turns=k)
                    for p in select(geo,g,2026):
                        self.assertTrue(mask[tuple(p['pos'].T)].all());self.assertFalse(mask[tuple(p['bg'].T)].any())
                        self.assertTrue(((p['pos']>=8)&(p['pos']<120)).all())

    def test_valid_image_denominator_and_connected_empty_zero(self):
        pairs=[dict(pos=np.array([[0,0],[0,1]]),bg=np.array([[1,0],[1,1]])),dict(pos=np.array([[2,0],[2,1]]),bg=np.array([[3,0],[3,1]]))]
        z=torch.zeros((1,1,4,4),requires_grad=True)
        a,_=auxiliary(z,[pairs],'PAIR',[1]);b,_=auxiliary(z.repeat(2,1,1,1),[pairs,[]],'PAIR',[1,2])
        self.assertEqual(a.item(),b.item());zero,_=auxiliary(z,[[]],'PAIR',[1]);zero.backward();self.assertEqual(float(z.grad.abs().sum()),0.)

    def test_pair_shuffle_gradients_and_rng(self):
        pairs=[dict(pos=np.array([[0,0]]),bg=np.array([[0,1]])),dict(pos=np.array([[1,0]]),bg=np.array([[1,1]]))]
        z=torch.tensor([[[[3.,1.],[-2.,-1.]]]],requires_grad=True)
        before=torch.get_rng_state().clone();a,_=auxiliary(z,[pairs],'PAIR',[9]);ga=torch.autograd.grad(a,z)[0]
        b,_=auxiliary(z,[pairs],'SHUFFLE',[9]);gb=torch.autograd.grad(b,z)[0]
        self.assertFalse(torch.equal(ga,gb));self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_scnp_matches_explicit_same_class_neighbors(self):
        torch.manual_seed(3);z=torch.randn(1,1,7,7,requires_grad=True);y=(torch.rand_like(z)>.5);f=torch.ones_like(y);f[:,:,0,:]=False
        out=scnp_logits(z,y,f)
        for row in range(7):
            for col in range(7):
                if not f[0,0,row,col]:self.assertEqual(out[0,0,row,col].item(),0.);continue
                vals=[z[0,0,i,j] for i in range(max(0,row-1),min(7,row+2)) for j in range(max(0,col-1),min(7,col+2)) if f[0,0,i,j] and y[0,0,i,j]==y[0,0,row,col]]
                expected=torch.stack(vals).min() if y[0,0,row,col] else torch.stack(vals).max()
                self.assertEqual(out[0,0,row,col].item(),expected.item())
        out.sum().backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_local_metric_perfect_cut_and_ties(self):
        from retinal_c1.evaluation import local
        s,label,fov=self.fixture();geo,_=build(label,fov,s,s)
        perfect,_=local(label.astype(float),geo,'synthetic')
        self.assertEqual(perfect['rank_inversion'],0.)
        self.assertEqual(perfect['path_pass_fraction'],1.)
        ties,_=local(np.full(label.shape,.5),geo,'synthetic')
        self.assertEqual(ties['rank_inversion'],.5)
        cut=label.astype(float);cut[:,30:35]=0
        broken,_=local(cut,geo,'synthetic')
        self.assertLess(broken['path_pass_fraction'],1.)
        self.assertGreater(broken['longest_missed_run'],0.)

    def test_small_plan_preserves_parent_draws(self):
        from retinal_c1.pilot import plan
        from retinal_c1.audit import OUT
        from retinal_release.common import read
        from retinal_m1.data import SkeletonDataset
        steps=plan();ds=SkeletonDataset('train',cache_size=0)
        self.assertEqual(len(steps),256)
        self.assertTrue(all(len(b)==4 for b in steps))
        self.assertEqual({ds.rows[d//4]['key'] for b in steps for e,d in b},set(read(OUT/'protocol.json')['training_32']))
        self.assertEqual(len({(e,d) for b in steps for e,d in b}),1024)

    def test_new_loader_matches_original_and_preserves_rng(self):
        from retinal_c1.pilot import Batches,plan
        from retinal_prefetch.loader import batch
        from retinal_m1.data import SkeletonDataset
        indices=plan()[0]
        reference,sig,_=batch(SkeletonDataset('train',cache_size=1),indices)
        before=torch.get_rng_state().clone()
        loader=Batches([],workers=1,depth=1)
        try:tensors,pairs,seeds,actual=loader._make(indices)
        finally:loader.close()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(reference,tensors)))
        for a,b in zip(sig,actual):self.assertTrue(all(a[k]==b[k] for k in a))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_foreign_chain_corridor_exclusion(self):
        from retinal_c1.evaluation import local
        s=np.zeros((144,144),bool);s[40,12:132]=True;s[45,12:132]=True
        label=binary_dilation(s,structure=disk(2));geo,_=build(label,np.ones_like(s),s,s)
        geo['_label']=label;geo['_chain_owner']=np.where(np.indices(s.shape)[0]<=42,0,1)
        result,rows=local(label.astype(float),geo,'synthetic')
        self.assertGreater(result['foreign_chain_paths_excluded'],0)
        self.assertEqual(result['rank_inversion'],0.)


if __name__=='__main__':unittest.main()
