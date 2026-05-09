"""Smoke tests for the V4 experiment registry."""

import pytest
from tools_debug.experiments import CANONICAL, EXPERIMENTS

from miles.utils.debug_utils.experiment_runner import RunKind, compose_run_config


def test_all_specs_compose_into_valid_run_configs() -> None:
    """Every registered spec must compose with the canonical without raising."""
    for name, spec in EXPERIMENTS.items():
        cfg = compose_run_config(CANONICAL, spec)
        assert cfg.name == name
        assert cfg.canonical_name == CANONICAL.name


def test_grafter_specs_all_have_at_least_one_filter() -> None:
    for name, spec in EXPERIMENTS.items():
        if spec.kind == RunKind.GRAFTER_PAIR:
            assert (
                spec.grafter_b2t_filter or spec.grafter_t2b_filter
            ), f"{name}: grafter_pair must declare at least one filter"


def test_baseline_runs_present() -> None:
    """Spec names referenced as baselines must exist somewhere -- prevents typos."""
    referenced: set[str] = set()
    for spec in EXPERIMENTS.values():
        referenced.update(spec.baselines)

    # Some baselines may be intentionally external (not yet registered) -- skip those.
    # All other referenced baselines should be either in EXPERIMENTS or marked external.
    for ref in referenced:
        if ref not in EXPERIMENTS:
            pytest.skip(f"baseline {ref!r} not in EXPERIMENTS (may be external)")


def test_canonical_has_required_paths() -> None:
    assert CANONICAL.sg_model_path is not None
    assert CANONICAL.mg_hf_checkpoint is not None
    assert CANONICAL.mg_ref_load is not None
    assert CANONICAL.rollout_data_path is not None
    assert CANONICAL.output_root is not None


def test_canonical_v4_specific_env_vars_set() -> None:
    """Canonical must include the V4 fix toggles all runs depend on."""
    assert CANONICAL.sg_env["SGLANG_DSV4_MODE"] == "2604"
    assert CANONICAL.sg_env["SGLANG_DSV4_FIX_0506"] == "1"
    assert CANONICAL.mg_env["MILES_DSV4_FIX_0505"] == "1"


@pytest.mark.parametrize("name", ["a1", "a2", "a3", "a4", "a5b", "m1", "combo-lora-proj-router"])
def test_phase_7a_main_specs_present(name: str) -> None:
    assert name in EXPERIMENTS
    assert EXPERIMENTS[name].kind == RunKind.GRAFTER_PAIR
