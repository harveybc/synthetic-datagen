#!/usr/bin/env python3
"""Regime-Clustered GAN Synthetic Data Generator (inspired by CLSGAN).

Pipeline:
1. Change-point detection (ruptures PELT) → split timeseries into segments
2. Extract features per segment → cluster into K regimes
3. Train lightweight GAN per regime on that regime's segments
4. Generate: sample regime sequence from transition matrix, generate per-regime
5. Reconstruct: stitch regime segments into continuous timeseries

Deterministic generation via seed control. Training uses GPU if available.
"""
import numpy as np
import pandas as pd
import json, os, time, warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path

warnings.filterwarnings("ignore")
_QUIET = os.environ.get("SDG_QUIET", "0") == "1"

# ── Change-Point Detection & Clustering ──────────────────────────────

def detect_regimes(prices: np.ndarray, n_regimes: int = 4,
                   min_segment: int = 30, penalty: float = 3.0
                   ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Detect regime segments via change-point detection + clustering.
    
    Returns:
        labels: array of regime labels per timestep
        segments: list of price segments
        switch_points: change-point indices
    """
    import ruptures as rpt
    from sklearn.cluster import AgglomerativeClustering

    # Work on log returns for stationarity
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))

    # PELT change-point detection
    algo = rpt.Pelt(model="rbf", jump=5, min_size=min_segment)
    algo.fit(log_ret.reshape(-1, 1))
    bkps = algo.predict(pen=penalty)

    # Build segments
    switch_points = [0] + bkps
    segments = []
    for i in range(len(switch_points) - 1):
        start, end = switch_points[i], switch_points[i + 1]
        segments.append(log_ret[start:end])

    if not _QUIET:
        print(f"  Change-points: {len(segments)} segments (penalty={penalty})")

    if len(segments) < n_regimes:
        # Not enough segments — lower penalty
        if not _QUIET:
            print(f"  Only {len(segments)} segments, reducing penalty...")
        penalty *= 0.5
        algo = rpt.Pelt(model="rbf", jump=5, min_size=min_segment)
        algo.fit(log_ret.reshape(-1, 1))
        bkps = algo.predict(pen=penalty)
        switch_points = [0] + bkps
        segments = []
        for i in range(len(switch_points) - 1):
            start, end = switch_points[i], switch_points[i + 1]
            segments.append(log_ret[start:end])

    # Extract features per segment for clustering
    features = []
    for seg in segments:
        if len(seg) < 2:
            features.append([0, 0, 0, 0, len(seg)])
            continue
        features.append([
            seg.mean(),           # trend
            seg.std(),            # volatility
            float(pd.Series(seg).skew()) if len(seg) > 2 else 0,  # asymmetry
            float(np.corrcoef(seg[:-1], seg[1:])[0, 1]) if len(seg) > 2 else 0,  # autocorr
            np.log(len(seg)),     # duration (log-scaled)
        ])
    features = np.array(features)
    features = np.nan_to_num(features, 0)

    # Normalize features
    f_std = features.std(axis=0) + 1e-10
    features_norm = (features - features.mean(axis=0)) / f_std

    # Cluster
    n_clusters = min(n_regimes, len(segments))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = clustering.fit_predict(features_norm)

    # Build per-timestep labels
    labels = np.zeros(len(log_ret), dtype=int)
    for i, (seg, lbl) in enumerate(zip(segments, cluster_labels)):
        start = switch_points[i]
        end = switch_points[i + 1]
        labels[start:end] = lbl

    # Transition matrix
    transitions = np.zeros((n_clusters, n_clusters))
    for i in range(len(cluster_labels) - 1):
        transitions[cluster_labels[i], cluster_labels[i + 1]] += 1
    # Normalize rows (add smoothing)
    transitions += 0.01
    transitions /= transitions.sum(axis=1, keepdims=True)

    if not _QUIET:
        for k in range(n_clusters):
            mask = cluster_labels == k
            n_segs = mask.sum()
            total_pts = sum(len(segments[i]) for i in range(len(segments)) if cluster_labels[i] == k)
            avg_ret = np.mean([segments[i].mean() for i in range(len(segments)) if cluster_labels[i] == k])
            avg_vol = np.mean([segments[i].std() for i in range(len(segments)) if cluster_labels[i] == k])
            print(f"  Regime {k}: {n_segs} segs, {total_pts} pts, μ={avg_ret:.6f}, σ={avg_vol:.6f}")

    return labels, segments, cluster_labels, transitions, switch_points


# ── Per-Regime GAN ───────────────────────────────────────────────────

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Simple 1D TCN generator for return sequences."""
    def __init__(self, z_dim: int = 8, hidden: int = 64, seq_len: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, seq_len: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_regime_gan(returns: np.ndarray, seq_len: int = 64, z_dim: int = 8,
                     hidden: int = 64, epochs: int = 200, batch_size: int = 32,
                     lr: float = 2e-4, device: str = "cpu") -> Generator:
    """Train a simple GAN on return sequences from one regime."""
    # Prepare sliding windows
    windows = []
    for i in range(0, len(returns) - seq_len + 1, seq_len // 2):
        windows.append(returns[i:i + seq_len])
    if len(windows) < 4:
        # Not enough data — use overlapping windows
        for i in range(0, len(returns) - seq_len + 1):
            windows.append(returns[i:i + seq_len])
    if len(windows) == 0:
        # Regime too short — return None, will use parametric fallback
        return None

    data = torch.FloatTensor(np.array(windows)).to(device)
    # Normalize to [-1, 1] range for GAN stability
    data_std = data.std() + 1e-10
    data_mean = data.mean()
    data_norm = (data - data_mean) / data_std

    G = Generator(z_dim, hidden, seq_len).to(device)
    D = Discriminator(seq_len, hidden).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        idx = torch.randint(0, len(data_norm), (min(batch_size, len(data_norm)),))
        real = data_norm[idx]

        # Train D
        z = torch.randn(len(real), z_dim, device=device)
        fake = G(z).detach()
        d_real = D(real)
        d_fake = D(fake)
        loss_D = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train G
        z = torch.randn(len(real), z_dim, device=device)
        fake = G(z)
        d_fake = D(fake)
        loss_G = criterion(d_fake, torch.ones_like(d_fake))
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    # Store normalization params on generator
    G.data_mean = float(data_mean)
    G.data_std = float(data_std)
    G.z_dim = z_dim
    G.eval()
    return G


# ── Full Model ───────────────────────────────────────────────────────

@dataclass
class RegimeGANModel:
    """Complete regime-clustered GAN model."""
    n_regimes: int
    transitions: np.ndarray
    regime_stats: List[Dict]   # per-regime: mean, std, count
    generators: List           # per-regime: trained Generator or None
    seq_len: int
    z_dim: int

    def save(self, path: str):
        """Save model (transitions + stats + generator weights)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_regimes": self.n_regimes,
            "transitions": self.transitions.tolist(),
            "regime_stats": self.regime_stats,
            "seq_len": self.seq_len,
            "z_dim": self.z_dim,
        }
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        for k, gen in enumerate(self.generators):
            if gen is not None:
                torch.save({
                    "state_dict": gen.state_dict(),
                    "data_mean": gen.data_mean,
                    "data_std": gen.data_std,
                }, out / f"gen_regime_{k}.pt")

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        generators = []
        for k in range(meta["n_regimes"]):
            pt = p / f"gen_regime_{k}.pt"
            if pt.exists():
                ckpt = torch.load(pt, map_location=device, weights_only=True)
                gen = Generator(meta["z_dim"], 64, meta["seq_len"]).to(device)
                gen.load_state_dict(ckpt["state_dict"])
                gen.data_mean = ckpt["data_mean"]
                gen.data_std = ckpt["data_std"]
                gen.z_dim = meta["z_dim"]
                gen.eval()
                generators.append(gen)
            else:
                generators.append(None)
        return cls(
            n_regimes=meta["n_regimes"],
            transitions=np.array(meta["transitions"]),
            regime_stats=meta["regime_stats"],
            generators=generators,
            seq_len=meta["seq_len"],
            z_dim=meta["z_dim"],
        )


def fit(prices: np.ndarray, n_regimes: int = 4, seq_len: int = 64,
        z_dim: int = 8, gan_epochs: int = 200, device: str = "cpu",
        penalty: float = 3.0) -> RegimeGANModel:
    """Full pipeline: detect regimes, train per-regime GANs."""
    if not _QUIET:
        print("Step 1: Detecting regimes...")
    labels, segments, cluster_labels, transitions, switch_pts = detect_regimes(
        prices, n_regimes=n_regimes, penalty=penalty
    )

    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))

    # Collect per-regime returns
    regime_returns = {k: [] for k in range(n_regimes)}
    for seg, lbl in zip(segments, cluster_labels):
        regime_returns[lbl].append(seg)

    if not _QUIET:
        print(f"\nStep 2: Training {n_regimes} per-regime GANs...")

    generators = []
    regime_stats = []
    for k in range(n_regimes):
        all_rets = np.concatenate(regime_returns[k]) if regime_returns[k] else np.array([])
        stats = {
            "mean": float(all_rets.mean()) if len(all_rets) > 0 else 0.0,
            "std": float(all_rets.std()) if len(all_rets) > 0 else 0.001,
            "count": int(len(all_rets)),
        }
        regime_stats.append(stats)

        if len(all_rets) < seq_len * 2:
            if not _QUIET:
                print(f"  Regime {k}: {len(all_rets)} pts — too few, parametric fallback")
            generators.append(None)
            continue

        if not _QUIET:
            print(f"  Regime {k}: {len(all_rets)} pts, training GAN...", end="", flush=True)
        t0 = time.time()
        gen = train_regime_gan(all_rets, seq_len=seq_len, z_dim=z_dim,
                               epochs=gan_epochs, device=device)
        if not _QUIET:
            print(f" {time.time()-t0:.1f}s")
        generators.append(gen)

    return RegimeGANModel(
        n_regimes=n_regimes,
        transitions=transitions,
        regime_stats=regime_stats,
        generators=generators,
        seq_len=seq_len,
        z_dim=z_dim,
    )


def generate(model: RegimeGANModel, n_samples: int, seed: int,
             initial_price: float = 1.0, device: str = "cpu") -> np.ndarray:
    """Generate synthetic prices from fitted regime GAN model. Deterministic via seed."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Sample regime sequence
    # First determine how many segments and their regimes
    generated_returns = []
    remaining = n_samples
    # Start from random regime weighted by how common each is
    counts = np.array([s["count"] for s in model.regime_stats], dtype=float)
    start_probs = counts / counts.sum() if counts.sum() > 0 else np.ones(model.n_regimes) / model.n_regimes
    current_regime = rng.choice(model.n_regimes, p=start_probs)

    while remaining > 0:
        chunk_len = min(model.seq_len, remaining)

        gen = model.generators[current_regime]
        if gen is not None and chunk_len == model.seq_len:
            # Use GAN
            z = torch.randn(1, gen.z_dim, device=device)
            with torch.no_grad():
                fake = gen(z).cpu().numpy().flatten()
            # Denormalize
            returns = fake * gen.data_std + gen.data_mean
        else:
            # Parametric fallback
            stats = model.regime_stats[current_regime]
            returns = rng.normal(stats["mean"], stats["std"], chunk_len)

        generated_returns.append(returns[:chunk_len])
        remaining -= chunk_len

        # Transition to next regime
        current_regime = rng.choice(model.n_regimes, p=model.transitions[current_regime])

    all_returns = np.concatenate(generated_returns)[:n_samples]

    # Convert to prices
    log_prices = np.log(initial_price) + np.cumsum(all_returns)
    return np.exp(log_prices)
