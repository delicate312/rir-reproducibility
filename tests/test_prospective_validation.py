import json
from pathlib import Path
import tempfile
import unittest
from experiments.ego_immune_recognition.riskbench_metadata.prospective_validation import admit, digest, family_fold, verify, write_json

class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.bundle=self.root/'bundle';self.bundle.mkdir()
        write_json(self.bundle/'prior_data_inventory.json',dict(raw_records=[dict(family='collision:old',base_scenario_id='old',state_sha256='oldhash')],historical_index_records=[]))
        write_json(self.bundle/'protocol.json',dict(created_utc='2026-09-08T00:00:00+00:00'))
        write_json(self.bundle/'manifest.json',dict(files={p.name:digest(p) for p in self.bundle.iterdir()}))
        (self.bundle/'MANIFEST.sha256').write_text(digest(self.bundle/'manifest.json'))
        self.state=self.root/'state.json';self.state.write_text('{"example": 1}')
        self.payload=dict(collection_started_utc='2026-09-09T00:00:00+00:00',collector='test collector',provenance_description='new scripts',model_scores_not_used=True,labels_sealed_before_scoring=True,new_parent_families=True,identity_review_completed=True,records=[dict(canonical_family_id='new:a',parent_script_id='a',state_path=str(self.state),state_sha256=digest(self.state))])
    def call(self):
        p=self.root/'candidate.json';write_json(p,self.payload)
        return admit(self.bundle,p,self.root/'admission.json')
    def test_accepts_new_identity_and_routes_deterministically(self):
        self.assertEqual(self.call(),1);self.assertEqual(family_fold('new:a'),family_fold('new:a'))
    def test_detects_frozen_artifact_tamper(self):
        (self.bundle/'protocol.json').write_text('{}')
        with self.assertRaises(ValueError):verify(self.bundle)
    def test_external_digest_mismatch_rejected(self):
        with self.assertRaises(ValueError):verify(self.bundle,'wrong')
    def test_renamed_old_parent_rejected(self):
        self.payload['records'][0]['parent_script_id']='old'
        with self.assertRaises(ValueError):self.call()
    def test_duplicate_family_rejected(self):
        self.payload['records']*=2
        with self.assertRaises(ValueError):self.call()
    def test_changed_raw_state_rejected(self):
        self.state.write_text('{}')
        with self.assertRaises(ValueError):self.call()
    def test_collection_before_freeze_rejected(self):
        self.payload['collection_started_utc']='2026-09-07T00:00:00+00:00'
        with self.assertRaises(ValueError):self.call()
    def test_missing_blinding_attestation_rejected(self):
        self.payload['model_scores_not_used']=False
        with self.assertRaises(ValueError):self.call()
    def test_renamed_duplicate_bytes_rejected(self):
        self.payload['records'].append(dict(self.payload['records'][0],canonical_family_id='new:b',parent_script_id='b'))
        with self.assertRaises(ValueError):self.call()

if __name__=='__main__':unittest.main()
