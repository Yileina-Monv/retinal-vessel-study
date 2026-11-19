import json, math, random
import numpy as np
import torch
from aux_loss import sparse_aux,total_loss,choose_groups,map_points,inverse_points

def run_tests():
    rows=[]
    def check(name, actual, expected=True, tol=1e-6):
        good = actual == expected if isinstance(expected,(bool,str)) else abs(actual-expected)<=tol
        rows.append({'name':name,'actual':actual,'expected':expected,'passed':bool(good)})
        if not good: raise AssertionError(name)
    group={'positive':[0,0],'negatives':[[0,1]]}
    x=torch.zeros((2,2,2,3),device='cpu',requires_grad=True)
    l=sparse_aux(x,[[group],[]]);check('full B empty crop normalization',l.item(),math.log(2)/2)
    l.backward();check('positive gradient',x.grad[0,1,0,0].item(),-.125)
    check('negative gradient',x.grad[0,1,0,1].item(),.125)
    check('background logit opposite',x.grad[0,0,0,0].item(),.125)
    check('empty crop gradient zero',x.grad[1].abs().sum().item(),0.)
    y=torch.zeros((1,2,2,3),requires_grad=True)
    two={**group,'negatives':[[0,1],[0,2]]}
    sparse_aux(y,[[two]]).backward()
    check('two negatives half total',y.grad[0,1,0,1:].sum().item(),.25)
    check('two equal negative weights',y.grad[0,1,0,1].item(),.125)
    for value in [-1000.,1000.]:
        a=torch.full((1,2,2,3),value,requires_grad=True);a.data[:,0]=-value
        b=sparse_aux(a,[[group]]);b.backward()
        check('extreme finite '+str(value),bool(torch.isfinite(b)&torch.isfinite(a.grad).all()))
    a=torch.ones((1,2,2,3),requires_grad=True);empty=sparse_aux(a,[[]]);empty.backward()
    check('all empty differentiable zero',empty.item(),0.);check('all empty grad zero',a.grad.abs().sum().item(),0.)
    base=torch.tensor(2.,requires_grad=True)
    check('lambda0 bypass metadata',total_loss(base,None,object(),0) is base)
    for name,g in [('no negative',{'positive':[0,0],'negatives':[]}),('duplicate negatives',{'positive':[0,0],'negatives':[[0,1],[0,1]]}),('polarity conflict',{'positive':[0,0],'negatives':[[0,0]]})]:
        try:sparse_aux(torch.zeros((1,2,2,3)),[[g]])
        except ValueError:check(name+' rejected',True)
        else:check(name+' rejected',False)
    # A shared negative accumulates precisely the two per-group weights.
    a=torch.zeros((1,2,2,3),requires_grad=True)
    sparse_aux(a,[[group,{'positive':[1,0],'negatives':[[0,1]]}]]).backward()
    check('shared negative accumulation',a.grad[0,1,0,1].item(),.25)
    # Analytic one-logit CE agrees with two-class CE and an independent scalar expression.
    a=torch.tensor([[[[.2,-.3]],[[.8,.5]]]],requires_grad=True)
    loss=sparse_aux(a,[[group]])
    import torch.nn.functional as F
    expected=.5*F.cross_entropy(a[:,:,0,0],torch.tensor([1]))+.5*F.cross_entropy(a[:,:,0,1],torch.tensor([0]))
    check('two-logit CE equivalence',loss.item(),expected.item())
    loss.backward(); analytic=a.grad.clone();eps=.001
    for idx in [(0,0,0,0),(0,1,0,0),(0,0,0,1),(0,1,0,1)]:
        ap=a.detach().clone();am=a.detach().clone();ap[idx]+=eps;am[idx]-=eps
        fd=((sparse_aux(ap,[[group]])-sparse_aux(am,[[group]]))/(2*eps)).item()
        check('finite difference '+str(idx),fd,analytic[idx].item(),8e-5)
    w=torch.tensor([[.1,-.2],[.3,.4]],requires_grad=True)
    features=torch.tensor([[1.,2.],[-.3,.6]])
    logits=(w@features).reshape(1,2,1,2)
    chain=sparse_aux(logits,[[group]]);chain.backward()
    analytic=w.grad.clone()
    for i in range(2):
        for j in range(2):
            wp=w.detach().clone();wm=w.detach().clone();wp[i,j]+=eps;wm[i,j]-=eps
            vp=sparse_aux((wp@features).reshape(1,2,1,2),[[group]])
            vm=sparse_aux((wm@features).reshape(1,2,1,2),[[group]])
            check('synthetic parameter chain '+str((i,j)),((vp-vm)/(2*eps)).item(),analytic[i,j].item(),8e-5)
    before=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
    c=[{**group,'native_y':i,'native_x':0} for i in range(30)]
    first=choose_groups(c,2027,1,0);second=choose_groups(c,2027,1,0)
    check('private RNG deterministic',first==second);check('all eligible targets retained',len(first),len(c))
    check('python RNG unchanged',before[0]==random.getstate())
    after=np.random.get_state();check('numpy RNG unchanged',all(np.array_equal(a,b) for a,b in zip(before[1],after)))
    check('torch CPU RNG unchanged',torch.equal(before[2],torch.random.get_rng_state()))
    # Independent grid construction tests include crop/padding, rotation/scale, mirrors and center convention.
    for A in [None,[[1.,0.],[0.,1.]],[[0.,-1.],[1.,0.]],[[.8,0.],[0.,.8]]]:
        for axes in [[],[0],[1],[0,1]]:
            kw=dict(bbox_lbs=[0,-2,-3],input_shape=[12,14],output_shape=[8,10],center=[6.,7.],affine=A,axes=axes)
            coords=np.array([[2.,4.],[3.,5.]])
            pixel,f=map_points(coords,**kw);back=inverse_points(f,**kw)
            check('continuous mapping inverse '+str(A)+str(axes),float(np.max(np.abs(back-coords))),0.,1e-12)
    # Compare native SpatialTransform's actual output grid, without generating random parameters.
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    yy,xx=torch.meshgrid(torch.arange(12),torch.arange(14),indexing='ij')
    identity=torch.stack([yy,xx]).float()
    for A in [None,np.eye(2),np.array([[.8,-.2],[.2,.8]])]:
        st=SpatialTransform((8,10),0,False,align_corners=False)
        params={'center_location_in_pixels':[6.,7.],'grid':None,'affine':A,'elastic_offsets':None}
        if A is not None:
            from batchgeneratorsv2.transforms.spatial.spatial import _create_centered_identity_grid2,_convert_my_grid_to_grid_sample_grid
            grid=_create_centered_identity_grid2((8,10)).float()@torch.from_numpy(A.T).float()
            params['grid']=_convert_my_grid_to_grid_sample_grid(grid,(12,14))
        out=st._apply_to_image(identity,**params)
        for axes in [[],[0,1]]:
            img=MirrorTransform((0,1))._apply_to_image(out,axes=axes)
            q=np.array([[2,3],[4,5]])
            inv=inverse_points(q,[0,0,0],[12,14],[8,10],[6.,7.],A,axes)
            actual=img[:,q[:,0],q[:,1]].numpy().T
            check('native grid coordinates '+str(A)+str(axes),float(np.max(abs(actual-inv))),0.,2e-6)
    return {'status':'PASS','count':len(rows),'tests':rows,'device':'CPU synthetic tensors only','real_network_gradients':False,'optimizer_updates':0,'GPU_queries':0}

if __name__=='__main__':print(json.dumps(run_tests(),indent=2))
