import collections
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze import estimates
from score import combine


class ReassessmentTests(unittest.TestCase):
    def test_domain_balance_does_not_overweight_larger_domain(self):
        raw = {'0': np.array([[0., 2.]]), '1': np.array([[2., 4.]]), '2': np.array([[10., 12.]])}
        checkpoints = [{'index': 0, 'domain': 'a'}, {'index': 1, 'domain': 'a'}, {'index': 2, 'domain': 'b'}]
        balanced, domains = combine(raw, checkpoints)
        np.testing.assert_array_equal(domains['a'], [[1., 3.]])
        np.testing.assert_array_equal(balanced, [[5.5, 7.5]])

    def test_group_bootstrap_matches_duplicated_sklearn_with_ties(self):
        records = [{'category': 'a', 'group_id': 'x', 'natural': [0.], 'generated': [1.]},
                   {'category': 'a', 'group_id': 'x', 'natural': [1.], 'generated': [1.]},
                   {'category': 'a', 'group_id': 'y', 'natural': [2.], 'generated': [0.]}]
        samples = [collections.Counter({'x': 1, 'y': 1}), collections.Counter({'x': 2, 'y': 1})]
        result = estimates(records, samples, {'a': 1}, 1)
        for j, sample in enumerate(samples):
            selected = [r for r in records for _ in range(sample[r['group_id']])]
            expected = roc_auc_score([0] * len(selected) + [1] * len(selected),
                                     [r['natural'][0] for r in selected] + [r['generated'][0] for r in selected])
            self.assertAlmostEqual(result[j, 0], expected)


if __name__ == '__main__':
    unittest.main()
