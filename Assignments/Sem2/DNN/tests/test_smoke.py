from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.data_pipeline import AdultDataset, load_adult_income_data


_SAMPLE_ROWS = """39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, >50K
37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, >50K
49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K
52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K
31, Private, 45781, Masters, 14, Never-married, Prof-specialty, Not-in-family, White, Female, 14084, 0, 50, United-States, >50K
42, Private, 159449, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 5178, 0, 40, United-States, >50K
"""


class SmokeTests(unittest.TestCase):
    def test_src_imports_and_data_contract(self) -> None:
        with patch("src.data_pipeline.requests.get") as mocked_get:
            response = Mock()
            response.text = _SAMPLE_ROWS
            response.raise_for_status = Mock()
            mocked_get.return_value = response

            data = load_adult_income_data(test_size=0.2, random_state=42)

        self.assertIsInstance(data, AdultDataset)
        self.assertEqual(data.n_samples, 10)
        self.assertEqual(data.n_features, 14)
        self.assertEqual(data.X_train.shape[0], 8)
        self.assertEqual(data.X_test.shape[0], 2)
        self.assertEqual(data.y_train.dtype, np.int64)
        self.assertEqual(data.y_test.dtype, np.int64)
        self.assertTrue(np.issubdtype(data.X_train.dtype, np.floating))
        self.assertTrue(np.issubdtype(data.X_test.dtype, np.floating))
        self.assertGreater(len(data.feature_names), data.n_features)


if __name__ == "__main__":
    unittest.main()

