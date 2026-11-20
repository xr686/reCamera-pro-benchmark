"""One-shot model demo for App Center. Results persist in the app data directory."""

import json
import os
import time
from pathlib import Path

from kit.app import App
from kit.config import app_dir_of, appdata_root
from kit.resources import ExternalNpuLease
from predict import model_inputs, prediction, save_prediction


class ModelDemo(App):
    id = json.loads(Path(__file__).with_name("model.json").read_text())["app_id"]
    owns_loop = True
    needs_frames = False
    needs_model = False

    def run(self):
        from rknnlite.api import RKNNLite

        root = Path(app_dir_of(self))
        spec = json.loads((root / "model.json").read_text())
        paths = [root / name for name in spec["inputs"]]
        inputs = model_inputs(spec, *paths)
        if os.environ.get("RECAMERA_NPU_MODE") != "legacy-direct":
            raise RuntimeError("Start the demo through App Center with an exclusive NPU allocation")
        lease = ExternalNpuLease()
        if lease.backend != "broker":
            raise RuntimeError("The firmware NPU broker is required")
        lease.acquire(timeout=10)
        try:
            lease.ready()
            runtime = RKNNLite()
            try:
                if runtime.load_rknn(str(root / "model.rknn")) or runtime.init_runtime():
                    raise RuntimeError(f"Could not initialize {spec['id']} on RV1126B")
                start = time.perf_counter_ns()
                outputs = runtime.inference(inputs=inputs, data_format=["nhwc" if t.ndim == 4 else "nchw" for t in inputs])
                latency = (time.perf_counter_ns() - start) / 1e6
                gallery = root / "gallery.npz"
                result = prediction(spec, outputs, gallery if gallery.exists() else None)
                info = dict(model=spec["id"], input="user" if paths else "generated", inference_ms=latency, result=result)
                public = save_prediction(info, Path(appdata_root()) / self.id)
            finally:
                runtime.release()
        finally:
            lease.release()
        message = dict(model=spec["id"], inference_ms=latency, file="prediction.json")
        if len(json.dumps(public["result"])) < 4096:
            message["prediction"] = public["result"]
        # Late result subscribers receive the cached prediction without another NPU call.
        while not self._stop_flag:
            self.emit(results=[message])
            time.sleep(1)
