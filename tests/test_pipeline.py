"""End-to-end ML artifact, scientific calibration and UI regression checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_transport_lab.data.generate import FEATURE_NAMES, generate_dataset, save_dataset
from tokamak_transport_lab.integration.config import run_config
from tokamak_transport_lab.surrogate.conformal import SplitConformal
from tokamak_transport_lab.transport.stiffness import chi_total


def test_all_five_features_have_the_declared_physical_meaning():
    data = generate_dataset(100, seed=21)
    x = data["X"]
    expected = chi_total(x[:, 0], **{k: x[:, i] for i, k in enumerate(FEATURE_NAMES) if i})
    np.testing.assert_allclose(data["y"], expected)
    point = np.array([7.0, 1.0, 3.0, 1.5, 0.01])
    base = float(chi_total(point[0], **dict(zip(FEATURE_NAMES[1:], point[1:], strict=True))))
    for i in range(5):
        trial = point.copy()
        trial[i] *= 1.1
        changed = float(
            chi_total(trial[0], **dict(zip(FEATURE_NAMES[1:], trial[1:], strict=True)))
        )
        assert changed != base


def test_conformal_uses_exact_order_statistic_without_interpolation():
    sc = SplitConformal().fit(np.arange(10.0), np.zeros(10), alpha=0.2)
    assert sc.q_hat == 8.0  # ceil(11*.8)=9th ordered score


def test_unattainable_small_sample_coverage_is_unbounded(tmp_path):
    sc = SplitConformal().fit([1.0, 2.0], [0.0, 0.0], alpha=0.01)
    assert np.isinf(sc.q_hat)
    path = tmp_path / "conformal.json"
    sc.save(path)
    assert "Infinity" not in path.read_text()
    assert np.isinf(SplitConformal.load(path).q_hat)


@pytest.mark.parametrize(
    "truth,pred,alpha", [([], [], 0.1), ([1], [1, 2], 0.1), ([np.nan], [0], 0.1), ([1], [1], 0)]
)
def test_invalid_conformal_data_raises(truth, pred, alpha):
    with pytest.raises(ValueError):
        SplitConformal().fit(truth, pred, alpha)


def test_train_save_load_and_adapter_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from tokamak_transport_lab.surrogate.bundle import TransportBundle
    from tokamak_transport_lab.surrogate.training import train_bundle
    from tokamak_transport_lab.transport.bundle_adapter import BundleAdapter

    train = generate_dataset(600, seed=1)
    val = generate_dataset(128, seed=2, sampling="iid")
    save_dataset(train, tmp_path / "train.npz")
    save_dataset(val, tmp_path / "validation.npz")
    trained = train_bundle(
        tmp_path / "train.npz",
        tmp_path / "validation.npz",
        tmp_path / "model",
        epochs=12,
        hidden=(16, 16),
    )
    loaded = TransportBundle.load(tmp_path / "model/model.pt")
    np.testing.assert_allclose(
        loaded.predict_chi(val["X"]), trained.predict_chi(val["X"]), rtol=1e-12
    )
    adapter = BundleAdapter(loaded)
    gradients = np.array([0.0, 2.0, 4.0, 6.0])
    x = np.column_stack(
        [gradients, np.full(4, 1.2), np.full(4, 3.0), np.full(4, 1.5), np.full(4, 0.02)]
    )
    np.testing.assert_allclose(adapter(gradients, chi_s=1.2, chi_neo=0.02), loaded.predict_chi(x))
    np.testing.assert_allclose(adapter(np.array([0.0, 2.0]), chi_neo=0.02), 0.02)
    threshold = adapter(np.array([3.0 - 1e-8, 3.0, 3.0 + 1e-8]), chi_s=1.2, chi_neo=0.02)
    assert np.max(threshold) - np.min(threshold) < 1e-6
    history = json.loads((tmp_path / "model/training.json").read_text())["histories"][0][
        "validation_loss"
    ]
    assert min(history[1:]) < history[0]
    cfg = {
        "model": {"type": "mlp", "path": str(tmp_path / "model/model.pt")},
        "grid": {"n_rho": 20},
    }
    result = run_config(cfg)
    assert result.metadata["converged"]
    assert result.metadata["transport_model"] == "mlp"
    assert result.metadata["artifact_sha256"] == loaded.fingerprint
    assert not result.metadata["used_fallback"]


def test_unknown_model_and_missing_artifact_never_fall_back():
    with pytest.raises(ValueError, match="Unknown transport"):
        run_config({"model": {"type": "misspelled"}})
    with pytest.raises(FileNotFoundError):
        run_config({"model": {"type": "ensemble", "path": "/tmp/ttl-nonexistent-model.pt"}})


def test_ensemble_band_propagates_each_member_and_contains_them(tmp_path):
    torch = pytest.importorskip("torch")
    from tokamak_transport_lab.integration.config import ensemble_profiles
    from tokamak_transport_lab.surrogate.bundle import TransportBundle
    from tokamak_transport_lab.surrogate.mlp import TransportMLP

    models = []
    for slope in (0.02, 0.025):
        m = TransportMLP(n_features=5, hidden=(), output_activation="linear")
        with torch.no_grad():
            m.net[0].weight.zero_()
            m.net[0].weight[0, 0] = slope
            m.net[0].bias.fill_(-3.0)
        models.append(m)
    b = TransportBundle(
        models,
        x_mean=np.zeros(5),
        x_std=np.ones(5),
        y_mean=0.0,
        y_std=1.0,
        bounds=np.array([[0, 20], [0.1, 5], [0, 6], [1, 2], [0.001, 0.1]]),
        hidden=(),
        seeds=[1, 2],
    )
    b.save(tmp_path / "ensemble.pt")
    cfg = {
        "model": {"type": "ensemble", "path": str(tmp_path / "ensemble.pt")},
        "grid": {"n_rho": 20},
        "source": {"S0_e": 500.0},
        "transport_e": {"a_over_LTe_crit": 1.0},
        "transport_i": {"a_over_LTe_crit": 1.0},
        "coupling": {"density": 1.0, "tau_eq": 1.0},
        "picard": {"tol": 1e-5, "max_iters": 100},
    }
    bands = ensemble_profiles(cfg)
    assert np.max(bands["te_upper"] - bands["te_lower"]) > 0.01
    assert np.all(bands["te_members"] >= bands["te_lower"])
    assert np.all(bands["te_members"] <= bands["te_upper"])
    np.testing.assert_allclose(bands["te_members"][:, -1], 500.0)
    assert "uncalibrated" in bands["label"]


def test_app_executes_and_identifies_stale_controls():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "src/tokamak_transport_lab/app/demo.py"),
        default_timeout=30,
    ).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["experiment"]["runs"][0][1]
    assert result.metadata["converged"]
    assert result.metadata["transport_model"] == "analytic"
    app.slider[0].set_value(700.0).run()
    assert any("previous experiment" in message.value for message in app.info)
    assert app.session_state["experiment"]["runs"][0][0]["source"]["S0_e"] == 650.0
    app.selectbox[1].select("mlp").run()
    app.text_input[0].set_value("/tmp/ttl-no-model.pt").run()
    app.button[0].click().run()
    assert app.error and "could not be completed" in app.error[0].value


@pytest.mark.parametrize(
    "cfg",
    [{"picrad": {}}, {"picard": {"tolerance": 1e-6}}, {"geometry": "unknown"}, {"qi_factor": -1}],
)
def test_bad_configuration_is_rejected(cfg):
    with pytest.raises(ValueError):
        run_config(cfg)


def test_analytic_app_imports_without_optional_torch_dependency():
    import subprocess
    import sys

    code = """
import sys
from importlib.abc import MetaPathFinder
class BlockTorch(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch deliberately unavailable")
sys.meta_path.insert(0, BlockTorch())
from tokamak_transport_lab.app.demo import main
from tokamak_transport_lab.surrogate.profile_uq import family_config
from tokamak_transport_lab.integration.config import run_config
assert run_config({"grid":{"n_rho":20}}).metadata["converged"]
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
