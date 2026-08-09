from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/run_stage4_failure_wiring_validation_v1.py")


def _validation_script():
    spec = importlib.util.spec_from_file_location("stage4_failure_wiring", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_failure_wiring_validation_uses_excluded_task_and_authorizes_attempt3() -> None:
    script = _validation_script()

    validation = script.run_validation()
    readiness = script.build_readiness(validation)

    assert validation["passed"] is True
    assert validation["official_benchmark_units_used"] == 0
    assert validation["provider_requests_made_by_this_script"] == 0
    assert validation["exclusion_manifest"]["task_count"] == 6
    assert validation["checks"]["provider_failure_materialized"] is True
    assert validation["checks"]["runner_classification_valid"] is True
    assert validation["checks"]["namespace_isolated"] is True
    assert readiness["attempt3_authorized"] is True
    assert readiness["attempt3_started"] is False
