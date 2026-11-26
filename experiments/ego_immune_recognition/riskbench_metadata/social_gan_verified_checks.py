"""Finite, deterministic validation and coordinate-adapter checks."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import social_gan_verified_retrain as r
from . import social_gan_verified_evaluation as e
from . import social_gan_clean_evaluation as old

def run():
 torch.set_num_threads(1);torch.manual_seed(123)
 # Real validation path must leave the training RNG unchanged.
 scene=np.zeros((50,3,2),np.float32);scene[:,1,0]=np.linspace(10,0,50);scene[:,2,1]=np.linspace(5,15,50)
 loader=DataLoader(r.SceneDataset([scene]),batch_size=1,collate_fn=r.collate_scenes)
 model,_=r.build_models(torch.device('cpu'));before=torch.random.get_rng_state().clone();a=r.validation_ade(model,loader,torch.device('cpu'),10);after=torch.random.get_rng_state();b=r.validation_ade(model,loader,torch.device('cpu'),10)
 assert torch.equal(before,after) and a==b and np.isfinite(a)
 # Deliberately drifting synthetic origin must not change fixed-origin risk.
 class Stub(torch.nn.Module):
  def __init__(self,drift):super().__init__();self.drift=drift
  def forward(self,obs,rel,seq):
   out=torch.zeros((30,obs.shape[1],2));out[:,seq[:,0],0]=self.drift;return out
 state={i:{'a':np.array([1.,0.])} for i in range(20)}
 kwargs=dict(fold=0,scenario_id='coordinate_check',samples=2,device=torch.device('cpu'))
 assert e.frame_actor_scores(Stub(0),state,19,**kwargs)==e.frame_actor_scores(Stub(10),state,19,**kwargs)
 assert old.frame_actor_scores(Stub(0),state,19,**kwargs)!=old.frame_actor_scores(Stub(10),state,19,**kwargs)
 assert e.frame_actor_scores(Stub(0),{},19,**kwargs)=={}
 print('PASS: validation RNG preservation, fixed-draw repeatability, fixed-origin risk invariance, empty history handling')
if __name__=='__main__':run()
