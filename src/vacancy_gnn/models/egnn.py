"""Dependency-light E(3)-equivariant GNN for per-arrangement energy.

The centerpiece model (PLAN.md Section 6). It is a small, self-contained
equivariant message-passing network in plain PyTorch: no ``e3nn`` or
``torch_geometric``, so the repo stays installable and CI-green with only the
optional ``[ml]`` extra (torch) added.

Design (PaiNN-style, orders l=0 and l=1):
- each node carries a scalar feature ``s`` (invariant) and a vector feature ``v``
  (equivariant, shape ``(3, F)``);
- a message block mixes a radial function of the edge distance (invariant) with
  the edge unit vector (equivariant), so scalar outputs are invariant and vector
  features rotate with the structure;
- an update block then feeds the vector features back into the scalars via
  rotation-invariant inner products, so the equivariant channel actually
  influences the predicted energy (without it the vector features would be dead
  weight and the model would collapse to a distance-only invariant network);
- the readout sums a per-node scalar energy contribution, giving a total energy
  invariant to global rotation, translation, and node permutation.

Torch is imported at module load, so this module must only be imported when the
``[ml]`` extra is installed. The rest of the package never imports it eagerly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from vacancy_gnn.data.featurize import Graph
from vacancy_gnn.models.reference import CompositionReference


def _graph_to_tensors(
    graph: Graph, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a numpy Graph into the tensors the network consumes.

    Vacancy markers arrive as nodes with :data:`VACANCY_MARKER_Z` (0), so they
    flow through the same embedding table as cations; no separate vacancy feature
    is needed.
    """
    z = torch.from_numpy(graph.node_z.astype(np.int64)).to(device)
    edge_index = torch.from_numpy(graph.edge_index.astype(np.int64)).to(device)
    edge_vec = torch.from_numpy(graph.edge_vec.astype(np.float32)).to(device)
    return z, edge_index, edge_vec


class RadialBasis(nn.Module):
    """Gaussian radial basis expansion of edge distances (invariant)."""

    centers: Tensor

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.width = float(centers[1] - centers[0]) if num_basis > 1 else cutoff

    def forward(self, dist: Tensor) -> Tensor:
        diff = dist[:, None] - self.centers[None, :]
        rbf = torch.exp(-((diff / self.width) ** 2))
        # Smooth cosine cutoff so contributions vanish at the boundary.
        envelope = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
        envelope = torch.where(dist <= self.cutoff, envelope, torch.zeros_like(dist))
        return rbf * envelope[:, None]


class EquivariantMessage(nn.Module):
    """One equivariant message-passing layer over scalar and vector features."""

    def __init__(self, hidden: int, num_basis: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )
        self.filter = nn.Linear(num_basis, 3 * hidden)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        rbf: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        dist = torch.linalg.norm(edge_vec, dim=1, keepdim=True).clamp_min(1e-8)
        unit = edge_vec / dist  # (E, 3), equivariant direction

        phi = self.scalar_mlp(s)[src] * self.filter(rbf)  # (E, 3H)
        ds, dv_scale, dvv = torch.split(phi, self.hidden, dim=1)

        # Scalar update: aggregate invariant scalar messages.
        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, dst, ds)

        # Vector update: scaled incoming vector features plus new vectors along the
        # edge direction (both transform equivariantly).
        vec_msg = dv_scale[:, None, :] * v[src] + dvv[:, None, :] * unit[:, :, None]
        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, dst, vec_msg)

        return s + agg_s, v + agg_v


class EquivariantUpdate(nn.Module):
    """PaiNN-style node update mixing vector features back into the scalars.

    Two learned linear maps ``U`` and ``V`` act on the feature axis of the ``(3,
    H)`` vector features. The scalar channel is then updated from the vector norm
    ``||Vv||`` and the rotation-invariant inner product ``<Uv, Vv>``, which is
    what lets the equivariant l=1 features influence an invariant energy. Vectors
    are updated by a scalar-gated multiple of ``Uv`` (still equivariant).
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.u_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
        )

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        uv = self.u_proj(v)  # (N, 3, H), equivariant
        vv = self.v_proj(v)  # (N, 3, H), equivariant
        vv_norm = torch.linalg.norm(vv, dim=1)  # (N, H), invariant
        inner = (uv * vv).sum(dim=1)  # (N, H), <Uv, Vv> invariant

        a = self.scalar_mlp(torch.cat([s, vv_norm, inner], dim=-1))
        ds, dv_gate = torch.split(a, self.hidden, dim=-1)

        return s + ds, v + dv_gate[:, None, :] * uv


class _EGNNNet(nn.Module):
    """The torch module: embeddings, message layers, and a scalar energy readout."""

    def __init__(
        self,
        hidden: int,
        num_layers: int,
        num_basis: int,
        cutoff: float,
        max_z: int = 100,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.embedding = nn.Embedding(max_z, hidden)
        self.rbf = RadialBasis(num_basis, cutoff)
        self.messages = nn.ModuleList(
            EquivariantMessage(hidden, num_basis) for _ in range(num_layers)
        )
        self.updates = nn.ModuleList(
            EquivariantUpdate(hidden) for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        s = self.embedding(z)
        v = torch.zeros(z.shape[0], 3, self.hidden, device=z.device)
        dist = torch.linalg.norm(edge_vec, dim=1)
        rbf = self.rbf(dist)
        for message, update in zip(self.messages, self.updates, strict=True):
            assert isinstance(message, EquivariantMessage)
            assert isinstance(update, EquivariantUpdate)
            s, v = message(s, v, edge_index, edge_vec, rbf)
            s, v = update(s, v)
        node_energy = self.readout(s).squeeze(-1)
        total: Tensor = node_energy.sum()
        return total


class EquivariantGNN:
    """Equivariant GNN energy model implementing :class:`EnergyModel`.

    Wraps the torch network with the numpy-facing ``fit``/``predict``/``save``/
    ``load`` interface used by the rest of the package.
    """

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 3,
        num_basis: int = 16,
        cutoff: float = 5.0,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
        reference_prior: NDArray[np.float64] | None = None,
        reference_shrinkage: float = 0.0,
    ) -> None:
        self.config = {
            "hidden": hidden,
            "num_layers": num_layers,
            "num_basis": num_basis,
            "cutoff": cutoff,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        self.device = torch.device(device)
        #: Optional composition-reference anchor and shrinkage strength (see
        #: :meth:`vacancy_gnn.models.reference.CompositionReference.fit` and
        #: IMPROVEMENTS.md P8); ``reference_shrinkage=0.0`` (the default)
        #: reproduces the plain unconstrained reference fit. Not part of
        #: ``config`` since it is not saved/loaded (see :meth:`save`).
        self.reference_prior = reference_prior
        self.reference_shrinkage = reference_shrinkage
        self._net: _EGNNNet | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._reference: CompositionReference | None = None

    def _build_net(self) -> _EGNNNet:
        torch.manual_seed(int(self.config["seed"]))
        return _EGNNNet(
            hidden=int(self.config["hidden"]),
            num_layers=int(self.config["num_layers"]),
            num_basis=int(self.config["num_basis"]),
            cutoff=float(self.config["cutoff"]),
        ).to(self.device)

    def fit(self, graphs: list[Graph], energies: NDArray[np.float64]) -> None:
        if len(graphs) == 0:
            raise ValueError("cannot fit on an empty dataset")
        y = np.asarray(energies, dtype=np.float64).ravel()
        if y.shape[0] != len(graphs):
            raise ValueError("number of energies must match number of graphs")

        # Learn only the per-arrangement residual over a per-species reference.
        self._reference = CompositionReference()
        self._reference.fit(
            graphs,
            y,
            prior=self.reference_prior,
            shrinkage=self.reference_shrinkage,
        )
        residual = y - self._reference.predict(graphs)

        self._target_mean = float(residual.mean())
        self._target_std = float(residual.std()) or 1.0
        targets = (residual - self._target_mean) / self._target_std

        self._net = self._build_net()
        opt = torch.optim.Adam(
            self._net.parameters(), lr=float(self.config["learning_rate"])
        )
        tensors = [_graph_to_tensors(g, self.device) for g in graphs]
        target_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

        self._net.train()
        for _ in range(int(self.config["epochs"])):
            opt.zero_grad()
            preds = torch.stack([self._net(*t) for t in tensors])
            loss = nn.functional.mse_loss(preds, target_t)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

    def predict(self, graphs: list[Graph]) -> NDArray[np.float64]:
        if self._net is None or self._reference is None:
            raise RuntimeError("model is not fitted; call fit() first")
        self._net.eval()
        with torch.no_grad():
            tensors = [_graph_to_tensors(g, self.device) for g in graphs]
            preds = torch.stack([self._net(*t) for t in tensors]).cpu().numpy()
        residual = preds * self._target_std + self._target_mean
        result: NDArray[np.float64] = residual + self._reference.predict(graphs)
        return result.astype(np.float64)

    def save(self, path: Path) -> None:
        if self._net is None or self._reference is None:
            raise RuntimeError("model is not fitted; call fit() first")
        # Weights in a sibling .pt file; config/normalization in the json.
        weights_path = path.with_suffix(".pt")
        torch.save(self._net.state_dict(), weights_path)
        payload = {
            "config": self.config,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "reference": self._reference.to_list(),
            "weights_file": weights_path.name,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> EquivariantGNN:
        payload = json.loads(path.read_text())
        model = cls(**payload["config"])
        model._target_mean = float(payload["target_mean"])
        model._target_std = float(payload["target_std"])
        model._reference = CompositionReference.from_list(payload["reference"])
        model._net = model._build_net()
        weights_path = path.with_name(payload["weights_file"])
        model._net.load_state_dict(torch.load(weights_path, map_location=model.device))
        return model
