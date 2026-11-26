"""Load the 15 frozen models; does not retrain or score dataset events."""
from pathlib import Path
import argparse,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from experiments.ego_immune_recognition.riskbench_metadata.prospective_validation import verify
from experiments.ego_immune_recognition.deepaccident.frozen_pilot import _load_model,_verify_frozen_runtime
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=root/'results/rir_prospective_freeze_20260908_v2');a=p.parse_args()
verify(a.bundle);_verify_frozen_runtime(a.bundle)
for seed in [20260812,20260813,20260814]:
 for fold in range(5):
  model,normalization,device=_load_model(a.bundle,seed,fold)
  assert len(normalization)>0
print('Frozen manifest, scoring-source hashes and all 15 checkpoint loads passed.')
