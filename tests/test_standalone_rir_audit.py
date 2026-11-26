"""Validate fast safe-only selection against the direct causal pipeline."""
import unittest
import numpy as np
from experiments.ego_immune_recognition.riskbench_metadata.standalone_rir_audit import vector_layout, vector_scores
from experiments.ego_immune_recognition.riskbench_metadata import damp_hard_safe_grid_selection as g


class StandaloneAuditTests(unittest.TestCase):
    def test_vector_scores_match_direct_with_heldout_refs_and_gaps(self):
        rng=np.random.default_rng(9)
        events={};components={};geometry={}
        for e in range(4):
            uid=f'e{e}'
            events[uid]=[];components[uid]={}
            for pos in range(8):
                events[uid].append(dict(frame_position=pos,frames_to_anchor=8-pos,aggregations={'inverse_ttc_ego_max':float(rng.uniform())}))
                components[uid][pos]={}
                geometry[(uid,pos)]={}
                for actor in ('a','b'):
                    geometry[(uid,pos)][actor]={'raw':float(rng.uniform())}
                    if actor=='a' and pos in (2,5):
                        continue
                    components[uid][pos][actor]={k:float(rng.uniform()) for k in ('stress_raw','damage_raw','persistence_raw','recovery_raw')}
        # A family with no pre-anchor observations stays in the denominator.
        events['empty']=[dict(frame_position=0,frames_to_anchor=40,aggregations={'inverse_ttc_ego_max':0.})]
        components['empty']={0:{}}
        geometry[('empty',0)]={}
        candidate=g.Candidate(.8,.25,.35,.4,.1,0)
        layout=vector_layout(events,components,geometry)
        train={"e0","e2"}
        calibration=g._calibration_from_components(components,train)
        refs={'component_calibration':calibration,
              'damp':np.sort([g._activation(v,calibration,candidate) for u in train for ts in components[u].values() for v in ts.values()]),
              'cpa':np.sort([v['raw'] for (u,p),ts in geometry.items() if u in train for v in ts.values()]),
              'physical':np.array([0.,1.])}
        slow=g._event_scores(event_uids=set(events),events=events,components=components,geometry=geometry,references=refs,candidate=candidate,allow_severe_cpa_bypass=True)
        fast=vector_scores(layout,[0,2],candidate)
        self.assertEqual(fast[-1],0.)
        np.testing.assert_allclose(fast,[slow[u]['relation_score'] for u in layout['uids']],atol=1e-12,rtol=0)
        # Selection scores must not depend on the TTC alarm reference or values.
        for fs in events.values():
            for frame in fs:
                frame['aggregations']['inverse_ttc_ego_max']=100000.
        changed=vector_layout(events,components,geometry)
        np.testing.assert_array_equal(fast,vector_scores(changed,[0,2],candidate))


if __name__=='__main__':
    unittest.main()
