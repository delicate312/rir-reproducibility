"""Numerical and causal invariants for the exploratory frozen-head audit."""
import unittest
import numpy as np

from experiments.ego_immune_recognition.riskbench_metadata.damp_incremental_value_audit import upper_tail_score, temporal_scores
from experiments.ego_immune_recognition.riskbench_metadata.damp_incremental_value_statistics import weighted_curve_metrics


class AuditInvariants(unittest.TestCase):
    def test_weighted_metrics_match_repeated_sample_sklearn(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([.1, .8, .8, .4, .4, .9])
        weights = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 1, 1, 2, 1]])
        result = weighted_curve_metrics(labels, scores, weights)
        for i, w in enumerate(weights):
            idx = np.repeat(np.arange(len(labels)), w)
            self.assertAlmostEqual(result['AP'][i], average_precision_score(labels[idx], scores[idx]))
            self.assertAlmostEqual(result['AUROC'][i], roc_auc_score(labels[idx], scores[idx]))
            self.assertAlmostEqual(result['pAUC10'][i], roc_auc_score(labels[idx], scores[idx], max_fpr=.1))

    def test_joint_tail_matches_brute_force_including_ties(self):
        ref = np.array([[.2, .1], [.2, .1], [.9, .1], [.7, .8], [1., 1.]])
        query = np.array([[.2, .1], [0., 0.], [1., 1.], [.8, .2], [.95, .95], [1.1, 1.]])
        expected = [1-(np.all(ref >= row, axis=1).sum()+1)/(len(ref)+1) for row in query]
        np.testing.assert_allclose(upper_tail_score(ref, query), expected)
        for row, result in zip(query, expected):
            self.assertAlmostEqual(upper_tail_score(ref, [row])[0], result)

    def test_same_actor_history_and_prefix_causality(self):
        rows = [("e", 0, "a"), ("e", 1, "b"), ("e", 2, "a"), ("e", 3, "a"), ("e", 4, "a")]
        values = np.array([.8, .99, .7, .9, .8])
        full = temporal_scores(rows, values, np.zeros(5), support=3)
        self.assertEqual(full[("e", 2)], 0.)
        self.assertAlmostEqual(full[("e", 4)], .8)
        prefix = temporal_scores(rows[:4], values[:4], np.zeros(4), support=3)
        self.assertEqual(prefix, {key: value for key, value in full.items() if key[1] < 4})
        gap = temporal_scores(rows, values, np.zeros(5), support=2, window=5)
        self.assertAlmostEqual(gap[("e", 2)], .75)
        # Actor b's .99 cannot contribute to actor a's temporal score.
        self.assertAlmostEqual(gap[("e", 4)], .85)


if __name__ == "__main__":
    unittest.main()
