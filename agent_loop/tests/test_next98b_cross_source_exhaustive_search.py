from __future__ import annotations

import inspect
import json

import pandas as pd

from src.next98b_cross_source_exhaustive_search import (
    build_exhaustive_candidate_specs,
    run_cross_source_exhaustive_search,
)


def test_exhaustive_union_keeps_all_unique_prior_specs_and_singles() -> None:
    scigen = pd.DataFrame(
        {
            "term_ids_json": [json.dumps(["a"]), json.dumps(["a", "b"])],
            "weights_json": [json.dumps([1.0]), json.dumps([1.0, 2.0])],
        }
    )
    wyformer = pd.DataFrame(
        {
            "term_ids": [["a", "b"], ["b"]],
            "weights": [[1.0, 2.0], [1.0]],
        }
    )
    terms = [{"term_id": "a"}, {"term_id": "b"}]
    specs = build_exhaustive_candidate_specs(
        scigen_records=scigen,
        wyformer_records=wyformer,
        eligible_terms=terms,
    )
    assert len(specs) == 3
    pair = next(spec for spec in specs if len(spec["term_ids"]) == 2)
    assert pair["origins"] == ["next87_complete", "next95_complete"]


def test_exhaustive_runner_has_no_validation_or_replication_input() -> None:
    parameters = inspect.signature(run_cross_source_exhaustive_search).parameters
    assert {"scigen_discovery_endpoint_dir", "wyformer_discovery_endpoint_dir"} <= set(
        parameters
    )
    assert not any("validation" in name or "replication" in name for name in parameters)
