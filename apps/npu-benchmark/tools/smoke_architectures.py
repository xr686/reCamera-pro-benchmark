"""Check empty/explicit model selection in the built app with a controlled lease."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def smoke(app):
    module = sys.modules[type(app).__module__]
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "npu-benchmark"
        (root / "architectures").mkdir(parents=True)
        (root / "architectures/suite.json").write_text(
            json.dumps({"models": {"starnet": {}, "gmlp": {}}})
        )
        with (
            patch.object(module, "appdata_root", return_value=tmp),
            patch.dict(os.environ, {"RECAMERA_NPU_MODE": "legacy-direct"}),
            patch.object(module, "ExternalNpuLease") as lease,
        ):
            lease.return_value.backend = "broker"
            default = SimpleNamespace(id="npu-benchmark", config={"model_ids": ""})
            type(app).prepare_runtime(default)
            assert default.selected == {"starnet", "gmlp"} and not default.records
            explicit = SimpleNamespace(
                id="npu-benchmark",
                config={"model_ids": "xcit", "benchmark_scope": "all_installed"},
            )
            type(app).prepare_runtime(explicit)
            assert explicit.selected == {"xcit"}
            with patch.object(
                module, "load_samples", return_value=("legacy", "sha", ["sample"])
            ) as load:
                legacy = SimpleNamespace(
                    id="npu-benchmark", config={"benchmark_scope": "all_installed"}
                )
                type(app).prepare_runtime(legacy)
                assert not legacy.selected and legacy.records == ["sample"]
                load.assert_called_once_with(root / "fixtures")
            (root / "architectures/suite.json").unlink()
            try:
                type(app).prepare_runtime(default)
            except ValueError as e:
                assert "architecture resource pack" in str(e)
            else:
                raise AssertionError(
                    "Missing resources must not start a legacy benchmark"
                )
