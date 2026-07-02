"""Tests for the equivariant GNN.

Skipped entirely when torch is not installed, so the core test suite stays green
without the optional ``[ml]`` extra. The headline tests assert the physical
symmetry the architecture exists to provide: the predicted energy is invariant to
global rotation, translation, and node permutation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vacancy_gnn.data.featurize import build_graph  # noqa: E402
from vacancy_gnn.data.synthetic import make_synthetic_dataset  # noqa: E402
from vacancy_gnn.models.base import EnergyModel  # noqa: E402
from vacancy_gnn.models.egnn import EquivariantGNN  # noqa: E402

from .conftest import ArrangementFactory  # noqa: E402


def _small_model(**kw: object) -> EquivariantGNN:
    params: dict[str, object] = {
        "hidden": 16,
        "num_layers": 2,
        "num_basis": 8,
        "epochs": 3,
        "seed": 0,
    }
    params.update(kw)
    return EquivariantGNN(**params)  # type: ignore[arg-type]


def _rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _transform(a: object, q: np.ndarray, shift: np.ndarray) -> object:
    """Rotate+translate cations, oxygen sublattice, and cell together."""
    return a.model_copy(  # type: ignore[attr-defined]
        update={
            "cation_positions": (a.positions_array() @ q.T + shift).tolist(),  # type: ignore[attr-defined]
            "oxygen_positions": (
                a.oxygen_positions_array() @ q.T + shift  # type: ignore[attr-defined]
            ).tolist(),
            "cell": (a.cell_array() @ q.T).tolist(),  # type: ignore[attr-defined]
        }
    )


def test_egnn_satisfies_protocol() -> None:
    assert isinstance(_small_model(), EnergyModel)


def test_predicted_energy_invariant_under_rotation(
    make_arrangement: ArrangementFactory,
) -> None:
    model = _small_model()
    ds = make_synthetic_dataset(n_compositions=4, seed=1)
    graphs = [build_graph(a, cutoff=5.0) for a in ds.arrangements]
    energies = np.array([a.energy_ev for a in ds.arrangements])
    model.fit(graphs, energies)

    a = make_arrangement("C", "FeMnAl", vacancy_sites=[0, 1], seed=5)
    a = a.model_copy(update={"cation_species": [26, 25, 13, 26]})
    g = build_graph(a, cutoff=5.0)

    q = _rotation(seed=2)
    shift = np.array([4.0, -1.0, 2.0])
    g_rot = build_graph(_transform(a, q, shift), cutoff=5.0)

    e = model.predict([g])[0]
    e_rot = model.predict([g_rot])[0]
    assert e == pytest.approx(e_rot, abs=1e-4)


def test_vector_channel_influences_energy(
    make_arrangement: ArrangementFactory,
) -> None:
    # The equivariant vector channel must actually feed the energy (the PaiNN
    # update block). Zeroing the vector features at every layer should change the
    # prediction; if it did not, the l=1 pathway would be dead weight and the model
    # a distance-only invariant net (the flaw this block fixes).
    from vacancy_gnn.models.egnn import _graph_to_tensors

    model = _small_model(epochs=30)
    ds = make_synthetic_dataset(n_compositions=6, seed=11)
    graphs = [build_graph(a, cutoff=5.0) for a in ds.arrangements]
    energies = np.array([a.energy_ev for a in ds.arrangements])
    model.fit(graphs, energies)
    assert model._net is not None

    a = make_arrangement("C", "FeMnAl", vacancy_sites=[0, 1], seed=6)
    a = a.model_copy(update={"cation_species": [26, 25, 13, 27]})
    tensors = _graph_to_tensors(build_graph(a, cutoff=5.0), model.device)

    model._net.eval()
    with torch.no_grad():
        full = float(model._net(*tensors))
        # Re-run with the vector features held at zero after every message layer.
        for update in model._net.updates:
            update.u_proj.weight.zero_()
            update.v_proj.weight.zero_()
        ablated = float(model._net(*tensors))
    assert abs(full - ablated) > 1e-4


def test_predicted_energy_invariant_under_permutation(
    make_arrangement: ArrangementFactory,
) -> None:
    model = _small_model()
    ds = make_synthetic_dataset(n_compositions=4, seed=3)
    graphs = [build_graph(a, cutoff=5.0) for a in ds.arrangements]
    energies = np.array([a.energy_ev for a in ds.arrangements])
    model.fit(graphs, energies)

    a = make_arrangement("C", "FeMnAl", vacancy_sites=[0], seed=6)
    a = a.model_copy(update={"cation_species": [26, 25, 13, 27]})
    g = build_graph(a, cutoff=5.0)

    perm = np.array([2, 0, 3, 1])
    a_p = a.model_copy(
        update={
            "cation_positions": a.positions_array()[perm].tolist(),
            "cation_species": a.species_array()[perm].tolist(),
        }
    )
    g_p = build_graph(a_p, cutoff=5.0)

    assert model.predict([g])[0] == pytest.approx(model.predict([g_p])[0], abs=1e-4)


def test_training_reduces_error() -> None:
    ds = make_synthetic_dataset(n_compositions=6, seed=4)
    graphs = [build_graph(a, cutoff=5.0) for a in ds.arrangements]
    energies = np.array([a.energy_ev for a in ds.arrangements])

    model = _small_model(epochs=1)
    model.fit(graphs, energies)
    err_before = np.abs(model.predict(graphs) - energies).mean()

    model_more = _small_model(epochs=80)
    model_more.fit(graphs, energies)
    err_after = np.abs(model_more.predict(graphs) - energies).mean()

    assert err_after < err_before


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        _small_model().predict([])


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _small_model().fit([], np.array([]))


def test_save_load_round_trip(tmp_path: Path) -> None:
    ds = make_synthetic_dataset(n_compositions=4, seed=7)
    graphs = [build_graph(a, cutoff=5.0) for a in ds.arrangements]
    energies = np.array([a.energy_ev for a in ds.arrangements])

    model = _small_model()
    model.fit(graphs, energies)
    before = model.predict(graphs)

    path = tmp_path / "egnn.json"
    model.save(path)
    restored = EquivariantGNN.load(path)
    after = restored.predict(graphs)

    np.testing.assert_allclose(before, after, atol=1e-5)
