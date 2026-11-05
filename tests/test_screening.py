"""Small data-independent checks for screening weights and feature preparation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common


class ScreeningTests(unittest.TestCase):
    def test_sparse_tfidf_minmax_matches_dense_including_constant_columns(self):
        values = np.array([[0, 2, 3, 0], [2, 4, 3, 0], [4, 6, 3, 0]], dtype=float)
        actual = common.minmax_scale_sparse_tfidf(csr_matrix(values))
        self.assertTrue(hasattr(actual, 'tocsr'))
        np.testing.assert_allclose(actual.toarray(), MinMaxScaler().fit_transform(values))

    def test_outcome_weights_match_config(self):
        np.testing.assert_allclose(
            common.adaptive_weights([True, False, True, False], [True, False, False, True]),
            [1.0, 0.75, 0.75, 5.0])

    def test_initial_validation_holds_records_out_and_falls_back(self):
        idx = [10, 13, 17, 20, 27, 31, 38, 42, 50, 61, 72, 83]
        reviewed = pd.DataFrame({'screening_Include': [1] * 6 + [0] * 6}, index=idx)
        positions = pd.Series(range(len(idx)), index=idx)
        matrix = csr_matrix(np.ones((len(idx), 2)))
        seen = []
        def mock_ensemble(train, held_out, features, positions, c_value, max_iter):
            self.assertTrue(set(train.index).isdisjoint(held_out))
            self.assertTrue((train['adaptive_weight'] == 1).all())
            seen.extend(held_out)
            return np.full(len(held_out), 0.25), np.zeros((len(held_out), 3)), []
        with patch.object(common, 'easy_ensemble_predict', side_effect=mock_ensemble):
            result = common.initial_validation_weights(reviewed, matrix, positions)
        self.assertEqual(set(seen), set(idx))
        np.testing.assert_allclose(result.loc[idx[:6], 'adaptive_weight'], 1.75)
        np.testing.assert_allclose(result.loc[idx[6:], 'adaptive_weight'], 1.25)
        small = reviewed.iloc[[0, 6, 7]]
        with patch.object(common, 'easy_ensemble_predict') as called:
            fallback = common.initial_validation_weights(small, matrix, positions)
        called.assert_not_called()
        self.assertTrue((fallback['adaptive_weight'] == 1).all())
        self.assertTrue(fallback['validation_probability'].isna().all())

    def test_normal_mode_passes_class_weights_to_all_models(self):
        n = 80
        labels = np.array([1] * 6 + [0] * (n - 6))
        df = pd.DataFrame({'totalscore': np.linspace(5, 0, n),
                           'screening_Include': labels,
                           'screening_Exclude': 1 - labels,
                           'FTscreening': np.where(labels == 1, 'Include', 'Exclude')})
        source = df[['FTscreening']].copy()
        positions = pd.Series(np.arange(n), index=df.index)
        matrix = csr_matrix(np.ones((n, 2)))
        seen = []
        class Model:
            classes_ = np.array([0, 1])
            n_iter_ = np.array([1])
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y, sample_weight):
                np.testing.assert_array_equal(sample_weight, np.where(y == 1, 2.0, 1.0))
                seen.append(len(y))
                return self
            def predict_proba(self, x):
                return np.full((x.shape[0], 2), 0.5)
        with patch.object(common, 'LogisticRegression', Model):
            result = common.run_screening(source, df, matrix, positions,
                                          mode='normal', max_rounds=1, plot=False)
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(result['round_metrics']), 1)


if __name__ == '__main__':
    unittest.main()
