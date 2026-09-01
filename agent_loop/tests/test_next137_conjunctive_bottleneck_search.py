import hashlib
import json

import numpy as np
import pandas as pd

import src.next129_coordination_protection as n129
import src.next133_compactness_protection as n133
import src.next137_conjunctive_bottleneck_search as n137


def test_materialize_bottleneck_features_uses_minimum_and_fails_open() -> None:
    table = pd.DataFrame(
        {
            n129.FEATURE_NAME: [n129.CLIP_NORMALIZED * 0.8, n129.CLIP_NORMALIZED * 0.5],
            n129.SUPPORT_COLUMN: [True, True],
            n133.PACKING_FEATURE: [n133.PACKING_CLIP * 0.4, n133.PACKING_CLIP],
            n133.PACKING_SUPPORT: [True, False],
            n133.VOLUME_FEATURE: [n133.VOLUME_CLIP, n133.VOLUME_CLIP * 0.25],
            n133.VOLUME_SUPPORT: [True, True],
        }
    )

    result = n137.materialize_bottleneck_features(table)

    assert np.allclose(result[n137.PACKING_MIN_FEATURE].iloc[:1], [0.4])
    assert np.isnan(result[n137.PACKING_MIN_FEATURE].iloc[1])
    assert result[n137.PACKING_MIN_SUPPORT].tolist() == [True, False]
    assert np.allclose(result[n137.VOLUME_MIN_FEATURE], [0.8, 0.25])
    assert result[n137.VOLUME_MIN_SUPPORT].tolist() == [True, True]


def test_frozen_bottleneck_configuration_universe_identity() -> None:
    configurations = n137.build_bottleneck_configurations()
    encoded = sorted(
        json.dumps(config, sort_keys=True, separators=(",", ":"))
        for config in configurations
    )

    assert len(configurations) == 49
    assert len({tuple(config["term_ids"] + config["weights"]) for config in configurations}) == 49
    assert hashlib.sha256("\n".join(encoded).encode()).hexdigest() == n137.EXPECTED_CONFIGURATION_SHA256
