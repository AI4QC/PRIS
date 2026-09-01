import pathlib
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from law_falsification import apply_law_set_unknown_fails  # noqa: E402


class StrictFalsePositiveSemanticsTests(unittest.TestCase):
    def test_any_missing_target_or_guard_feature_fails_closed(self):
        frame = pd.DataFrame(
            {
                "x": [0.5, np.nan, 0.5, 2.0],
                "g": [1.0, 1.0, np.nan, 0.0],
            }
        )
        rules = [
            {
                "description": "if g > 0.5 then x <= 1",
                "feature": "x",
                "family": "guarded-one-sided",
                "origin": "test",
                "side": "hi",
                "thresholds": [1.0],
                "guard_feature": "g",
                "guard_side": "hi",
                "guard_threshold": 0.5,
            }
        ]

        passed, known = apply_law_set_unknown_fails(frame, rules)

        np.testing.assert_array_equal(known, [True, False, False, True])
        np.testing.assert_array_equal(passed, [True, False, False, True])


if __name__ == "__main__":
    unittest.main()
