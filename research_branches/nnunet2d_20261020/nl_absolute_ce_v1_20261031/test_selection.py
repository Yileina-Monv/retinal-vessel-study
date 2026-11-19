"""Synthetic sidecar rejection/collision tests; no real images or model forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,numpy as np,torch
import replay

def main():
    torch.set_num_threads(1)
    cases=[]
    def run(name,targets,gt,expected):
        # Fill category registry so process_crop cannot create an actual-image atlas.
        replay.ATLAS={str(i):None for i in range(12)}
        replay.MAPPING={'SYNTHETIC':{'NL':targets}}
        replay.native=lambda _: {'GT':gt,'Omega':np.ones_like(gt,bool),'near':np.ones_like(gt,bool)}
        c=dict(image='SYNTHETIC',native_shape=list(gt.shape),bbox_lbs=[0,0,0],bbox_ubs=[1,8,8],
               input_shape=[8,8],output_shape=[8,8],center=[4.,4.],affine=None,axes=[])
        out={'image':torch.zeros((3,8,8)),'segmentation':[torch.from_numpy(gt[None].astype(np.int16))]}
        rec=replay.process_crop(c,None,out,2027,1,0)
        actual=[x['status'] for x in rec['targets']]
        if actual!=expected:raise AssertionError((name,actual,expected))
        cases.append({'name':name,'expected':expected,'actual':actual,'counts':rec['counts'],'passed':True})
    gt=np.zeros((8,8),bool);gt[2,2]=True
    point=lambda y,x,b:dict(y=y,x=x,background_coords=b)
    run('single negative',[point(2,2,[[2,4]])],gt,['SELECTED'])
    run('two negatives',[point(2,2,[[2,4],[3,4]])],gt,['SELECTED'])
    run('negative leaves output',[point(2,2,[[20,20]])],gt,['NO_NEGATIVE'])
    run('negative wrong polarity',[point(2,2,[[2,2]])],gt,['NO_NEGATIVE'])
    run('positive wrong polarity',[point(1,1,[[2,4]])],gt,['GT_DOMAIN_INVALID'])
    run('duplicate positive deterministic identity',[point(2,2,[[2,4]]),point(2,2,[[3,4]])],gt,['SELECTED','POSITIVE_COLLISION'])
    run('positive outside output',[point(20,20,[[2,4]])],gt,['OUTSIDE_OUTPUT'])
    run('no native target',[],gt,[])
    report={'status':'PASS','tests':cases,'count':len(cases),'actual_native_crop_instances':0,'GPU_queries':0,'model_forwards':0}
    p=replay.OUT/'cpu_selection_tests.json'
    with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('SYNTHETIC_SELECTION_TESTS',len(cases),'PASS')

if __name__=='__main__':main()
