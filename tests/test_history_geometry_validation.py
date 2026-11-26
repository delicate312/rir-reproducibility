"""Causality, matched-sample integrity and release-censoring checks."""
import unittest
import numpy as np
import pandas as pd

from experiments.ego_immune_recognition.riskbench_metadata.history_geometry_validation import (
    history_controls, match_snapshots, release_diagnostic,
)


class HistoryGeometryTests(unittest.TestCase):
    def test_small_sample_randomization_is_exact(self):
        from experiments.ego_immune_recognition.riskbench_metadata.history_geometry_report import paired_randomization_p
        rng = np.random.default_rng(1)
        self.assertEqual(paired_randomization_p([1., 1., 1., 0.], rng), .25)
        self.assertEqual(paired_randomization_p([0., 0.], rng), 1.)
        self.assertEqual(paired_randomization_p([1., -1.], rng), 1.)

    def test_prefix_causality_and_gap_reset(self):
        pos = [0, 1, 2, 5, 6, 7]
        c = [.2, .4, .8, .3, .9, .7]
        full = history_controls(pos, c)
        np.testing.assert_allclose(full[:4], history_controls(pos[:4], c[:4]))
        np.testing.assert_allclose(full[3:], history_controls(pos[3:], c[3:]))
        self.assertAlmostEqual(full[2][3], .8 + 10 * .3)

    def test_matching_unique_caliper_and_score_independence(self):
        def row(cp, px):
            return dict(cpa_raw=cp, px=px, py=0., vx=0., vy=0., tcpa=.5, dcpa=2., damp=.1)
        a = pd.DataFrame([row(.3, 0), row(.31, 0), row(.8, 100)])
        b = pd.DataFrame([row(.305, 0), row(.31, 1), row(.8, 0)])
        pairs = match_snapshots(a, b, "geometry")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({x[1] for x in pairs}), 2)
        a.damp = [.9, .1, .8]
        self.assertEqual(pairs, match_snapshots(a, b, "geometry"))

    def test_release_requires_full_followup_and_confirmation(self):
        fs = [dict(source_frame_index=i) for i in range(1, 31)]
        values = [1.] * 12 + [0.] * 18
        result = release_diagnostic(fs, values, .5, 10)
        self.assertEqual(result['complete'], 1)
        self.assertEqual(result['pre_alert'], 1)
        self.assertAlmostEqual(result['release_confirm_s'], .35)
        self.assertEqual(result['release_censored'], 0)
        self.assertEqual(release_diagnostic(fs[:-1], values[:-1], .5, 10)['complete'], 0)
        self.assertEqual(release_diagnostic(fs, [1.] * 30, .5, 10)['release_censored'], 1)


if __name__ == '__main__':
    unittest.main()
