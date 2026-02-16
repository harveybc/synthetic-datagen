"""
TimeGAN-based synthetic time series generator.

Based on: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)

Architecture:
1. Embedding network: maps real data to latent space
2. Recovery network: maps latent back to data space
3. Generator: produces synthetic latent sequences from noise
4. Discriminator: distinguishes real vs synthetic in latent space
5. Supervisor: autoregressive model in latent space (temporal consistency)

Training phases:
1. Autoencoder (embedding + recovery)
2. Supervised (supervisor in latent space)
3. Joint (all components together)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class TimeGANGenerator:
    """Plugin: TimeGAN synthetic typical_price generator."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {
            "window_size": 48,
            "latent_dim": 24,
            "hidden_dim": 24,
            "n_layers": 3,
            "epochs_ae": 200,
            "epochs_sup": 200,
            "epochs_joint": 200,
            "patience_ae": 30,
            "patience_sup": 30,
            "patience_joint": 50,
            "min_delta": 1e-6,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "use_returns": True,
            "start_datetime": "2020-01-01 00:00:00",
            "interval_hours": 4,
        }
        if config:
            self.cfg.update(config)
        self._models = None

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    def train(self, save_dir: Optional[str] = None) -> str:
        """Train TimeGAN on real data. Returns path to saved model directory."""
        import tensorflow as tf
        from tensorflow import keras
        from app.data_processor import (
            load_multiple_csv, prices_to_returns, create_windows,
        )

        cfg = self.cfg
        train_data = cfg.get("train_data")
        if not train_data:
            raise ValueError("train_data required")
        if isinstance(train_data, str):
            train_data = [train_data]

        df = load_multiple_csv(train_data)
        prices = df["typical_price"].values.astype(np.float64)

        if cfg.get("use_returns", True):
            series = prices_to_returns(prices)
        else:
            series = prices.copy()

        # Normalize to [0, 1] for stable GAN training
        self._data_min = series.min()
        self._data_max = series.max()
        self._data_range = self._data_max - self._data_min
        if self._data_range < 1e-10:
            self._data_range = 1.0
        normalized = (series - self._data_min) / self._data_range

        window_size = cfg["window_size"]
        windows = create_windows(normalized, window_size)
        # Reshape to (N, T, 1) for RNN
        all_X = windows[:, :, np.newaxis].astype(np.float32)

        # Chronological train/val split (last 15% as validation)
        val_frac = cfg.get("val_fraction", 0.15)
        n_total = len(all_X)
        n_val = max(1, int(n_total * val_frac))
        X = all_X[:-n_val]
        X_val = all_X[-n_val:]
        log.info(f"Training data: {X.shape[0]} windows, validation: {X_val.shape[0]} windows (size {window_size})")

        # Store metadata
        self._initial_price = float(prices[0])
        self._anchor_prices = prices.copy()

        # Build models
        hidden_dim = cfg["hidden_dim"]
        latent_dim = cfg["latent_dim"]
        n_layers = cfg["n_layers"]
        n_features = 1

        embedder, recovery, generator, supervisor, discriminator = \
            self._build_models(window_size, n_features, hidden_dim, latent_dim, n_layers)

        # Training
        lr = cfg["learning_rate"]
        batch_size = cfg["batch_size"]

        min_delta = cfg.get("min_delta", 1e-6)

        # Phase 1: Autoencoder
        log.info("Phase 1: Autoencoder training")
        ae_optimizer = keras.optimizers.Adam(lr)
        self._train_autoencoder(
            embedder, recovery, X, X_val, ae_optimizer,
            cfg["epochs_ae"], batch_size,
            patience=cfg.get("patience_ae", 30), min_delta=min_delta,
        )

        # Phase 2: Supervised
        log.info("Phase 2: Supervised training")
        sup_optimizer = keras.optimizers.Adam(lr)
        self._train_supervised(
            embedder, supervisor, X, X_val, sup_optimizer,
            cfg["epochs_sup"], batch_size,
            patience=cfg.get("patience_sup", 30), min_delta=min_delta,
        )

        # Phase 3: Joint training
        log.info("Phase 3: Joint training")
        g_optimizer = keras.optimizers.Adam(lr)
        d_optimizer = keras.optimizers.Adam(lr)
        self._train_joint(
            embedder, recovery, generator, supervisor, discriminator,
            X, X_val, g_optimizer, d_optimizer,
            cfg["epochs_joint"], batch_size, latent_dim,
            patience=cfg.get("patience_joint", 50), min_delta=min_delta,
        )

        self._models = {
            "embedder": embedder,
            "recovery": recovery,
            "generator": generator,
            "supervisor": supervisor,
            "discriminator": discriminator,
        }

        # Save
        save_dir = save_dir or cfg.get("save_model", "timegan_model")
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self._models.items():
            model.save(os.path.join(save_dir, f"{name}.keras"))
        meta = {
            "window_size": window_size,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "data_min": float(self._data_min),
            "data_max": float(self._data_max),
            "data_range": float(self._data_range),
            "initial_price": float(self._initial_price),
            "use_returns": cfg.get("use_returns", True),
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"TimeGAN saved to {save_dir}")
        return save_dir

    def load_model(self, model_dir: str) -> None:
        """Load pre-trained TimeGAN."""
        from tensorflow import keras

        with open(os.path.join(model_dir, "meta.json")) as f:
            self._meta = json.load(f)

        self._models = {}
        for name in ["generator", "supervisor", "recovery"]:
            path = os.path.join(model_dir, f"{name}.keras")
            self._models[name] = keras.models.load_model(path, compile=False)

        self._data_min = self._meta["data_min"]
        self._data_max = self._meta["data_max"]
        self._data_range = self._meta["data_range"]
        log.info(f"Loaded TimeGAN from {model_dir}")

    def generate(self, seed: int, n_samples: int,
                 start_datetime: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic data using trained TimeGAN."""
        if self._models is None:
            raise RuntimeError("No model loaded")

        generator = self._models["generator"]
        supervisor = self._models["supervisor"]
        recovery = self._models["recovery"]
        meta = self._meta if hasattr(self, '_meta') else {
            "window_size": self.cfg["window_size"],
            "latent_dim": self.cfg["latent_dim"],
            "data_min": self._data_min,
            "data_max": self._data_max,
            "data_range": self._data_range,
            "initial_price": self._initial_price,
            "use_returns": self.cfg.get("use_returns", True),
        }

        window_size = meta["window_size"]
        latent_dim = meta["latent_dim"]

        rng = np.random.default_rng(seed)
        n_windows = (n_samples // window_size) + 2

        # Generate: noise → generator → supervisor → recovery
        Z = rng.standard_normal((n_windows, window_size, latent_dim)).astype(np.float32)
        E_hat = generator.predict(Z, verbose=0)
        H_hat = supervisor.predict(E_hat, verbose=0)
        X_hat = recovery.predict(H_hat, verbose=0)

        # Denormalize
        synthetic_norm = X_hat[:, :, 0].reshape(-1)[:n_samples]
        synthetic = synthetic_norm * meta["data_range"] + meta["data_min"]

        if meta.get("use_returns", True):
            # Convert returns to prices with windowed reconstruction
            anchor_prices = getattr(self, '_anchor_prices', None)
            if anchor_prices is None:
                initial_price = meta.get("initial_price", 1.3)
                anchor_prices = np.array([initial_price])

            prices_list = []
            for w in range(n_windows):
                win_returns = X_hat[w, :, 0] * meta["data_range"] + meta["data_min"]
                win_returns = win_returns - win_returns.mean()
                anchor = anchor_prices[rng.integers(len(anchor_prices))]
                log_p = np.concatenate([[np.log(anchor)], win_returns])
                win_prices = np.exp(np.cumsum(log_p))
                prices_list.append(win_prices[1:])  # skip anchor
            synthetic_prices = np.concatenate(prices_list)[:n_samples]
        else:
            synthetic_prices = synthetic

        start_dt = pd.to_datetime(
            start_datetime or self.cfg.get("start_datetime", "2020-01-01 00:00:00")
        )
        interval = timedelta(hours=self.cfg.get("interval_hours", 4))
        dates = [start_dt + i * interval for i in range(n_samples)]

        return pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": synthetic_prices,
        })

    # ── Model building ──────────────────────────────────────────────────

    @staticmethod
    def _build_models(seq_len, n_features, hidden_dim, latent_dim, n_layers):
        from tensorflow import keras
        from tensorflow.keras import layers

        # Embedder: X → H
        e_input = keras.Input(shape=(seq_len, n_features), name="e_input")
        e = e_input
        for i in range(n_layers):
            e = layers.GRU(hidden_dim, return_sequences=True, name=f"e_gru_{i}")(e)
        e_output = layers.Dense(hidden_dim, activation="sigmoid", name="e_dense")(e)
        embedder = keras.Model(e_input, e_output, name="embedder")

        # Recovery: H → X_tilde
        r_input = keras.Input(shape=(seq_len, hidden_dim), name="r_input")
        r = r_input
        for i in range(n_layers):
            r = layers.GRU(hidden_dim, return_sequences=True, name=f"r_gru_{i}")(r)
        r_output = layers.Dense(n_features, name="r_dense")(r)
        recovery = keras.Model(r_input, r_output, name="recovery")

        # Generator: Z → E_hat
        g_input = keras.Input(shape=(seq_len, latent_dim), name="g_input")
        g = g_input
        for i in range(n_layers):
            g = layers.GRU(hidden_dim, return_sequences=True, name=f"g_gru_{i}")(g)
        g_output = layers.Dense(hidden_dim, activation="sigmoid", name="g_dense")(g)
        generator = keras.Model(g_input, g_output, name="generator")

        # Supervisor: H → H_hat (autoregressive in latent space)
        s_input = keras.Input(shape=(seq_len, hidden_dim), name="s_input")
        s = s_input
        for i in range(n_layers - 1):
            s = layers.GRU(hidden_dim, return_sequences=True, name=f"s_gru_{i}")(s)
        s_output = layers.Dense(hidden_dim, activation="sigmoid", name="s_dense")(s)
        supervisor = keras.Model(s_input, s_output, name="supervisor")

        # Discriminator: H → Y (real/fake)
        d_input = keras.Input(shape=(seq_len, hidden_dim), name="d_input")
        d = d_input
        for i in range(n_layers):
            d = layers.GRU(hidden_dim, return_sequences=True, name=f"d_gru_{i}")(d)
        d_output = layers.Dense(1, name="d_dense")(d)
        discriminator = keras.Model(d_input, d_output, name="discriminator")

        return embedder, recovery, generator, supervisor, discriminator

    # ── Training phases ─────────────────────────────────────────────────

    @staticmethod
    def _train_autoencoder(embedder, recovery, X, X_val, optimizer, epochs, batch_size,
                           patience=30, min_delta=1e-6):
        import tensorflow as tf
        n = len(X)
        steps = max(1, n // batch_size)
        best_val_loss = float("inf")
        wait = 0
        best_e_weights = None
        best_r_weights = None

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            total_loss = 0.0
            for step in range(steps):
                batch_idx = idx[step * batch_size : (step + 1) * batch_size]
                x_batch = X[batch_idx]

                with tf.GradientTape() as tape:
                    h = embedder(x_batch, training=True)
                    x_tilde = recovery(h, training=True)
                    loss = tf.reduce_mean(tf.abs(x_batch - x_tilde))

                trainable = embedder.trainable_variables + recovery.trainable_variables
                grads = tape.gradient(loss, trainable)
                optimizer.apply_gradients(zip(grads, trainable))
                total_loss += loss.numpy()

            # Validation loss
            h_val = embedder(X_val, training=False)
            x_val_tilde = recovery(h_val, training=False)
            val_loss = tf.reduce_mean(tf.abs(X_val - x_val_tilde)).numpy()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                wait = 0
                best_e_weights = [w.numpy().copy() for w in embedder.trainable_variables]
                best_r_weights = [w.numpy().copy() for w in recovery.trainable_variables]
            else:
                wait += 1

            if (epoch + 1) % 50 == 0:
                log.info(f"  AE epoch {epoch+1}/{epochs}: train={total_loss/steps:.6f}, val={val_loss:.6f} (best_val={best_val_loss:.6f}, wait={wait}/{patience})")

            if wait >= patience:
                log.info(f"  AE early stop at epoch {epoch+1}, best_val={best_val_loss:.6f}")
                break

        if best_e_weights is not None:
            for var, val in zip(embedder.trainable_variables, best_e_weights):
                var.assign(val)
            for var, val in zip(recovery.trainable_variables, best_r_weights):
                var.assign(val)

    @staticmethod
    def _train_supervised(embedder, supervisor, X, X_val, optimizer, epochs, batch_size,
                          patience=30, min_delta=1e-6):
        import tensorflow as tf
        n = len(X)
        steps = max(1, n // batch_size)
        best_val_loss = float("inf")
        wait = 0
        best_weights = None

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            total_loss = 0.0
            for step in range(steps):
                batch_idx = idx[step * batch_size : (step + 1) * batch_size]
                x_batch = X[batch_idx]

                with tf.GradientTape() as tape:
                    h = embedder(x_batch, training=False)
                    h_hat = supervisor(h, training=True)
                    loss = tf.reduce_mean(tf.abs(h[:, 1:, :] - h_hat[:, :-1, :]))

                grads = tape.gradient(loss, supervisor.trainable_variables)
                optimizer.apply_gradients(zip(grads, supervisor.trainable_variables))
                total_loss += loss.numpy()

            # Validation loss
            h_val = embedder(X_val, training=False)
            h_val_hat = supervisor(h_val, training=False)
            val_loss = tf.reduce_mean(tf.abs(h_val[:, 1:, :] - h_val_hat[:, :-1, :])).numpy()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                wait = 0
                best_weights = [w.numpy().copy() for w in supervisor.trainable_variables]
            else:
                wait += 1

            if (epoch + 1) % 50 == 0:
                log.info(f"  SUP epoch {epoch+1}/{epochs}: train={total_loss/steps:.6f}, val={val_loss:.6f} (best_val={best_val_loss:.6f}, wait={wait}/{patience})")

            if wait >= patience:
                log.info(f"  SUP early stop at epoch {epoch+1}, best_val={best_val_loss:.6f}")
                break

        if best_weights is not None:
            for var, val in zip(supervisor.trainable_variables, best_weights):
                var.assign(val)

    @staticmethod
    def _train_joint(embedder, recovery, generator, supervisor, discriminator,
                     X, X_val, g_optimizer, d_optimizer, epochs, batch_size, latent_dim,
                     patience=50, min_delta=1e-6):
        import tensorflow as tf
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        n = len(X)
        steps = max(1, n // batch_size)
        seq_len = X.shape[1]
        best_val_moment = float("inf")
        wait = 0
        best_weights = None

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            g_total, d_total, v_total = 0.0, 0.0, 0.0
            for step in range(steps):
                batch_idx = idx[step * batch_size : (step + 1) * batch_size]
                x_batch = X[batch_idx]
                bs = len(batch_idx)

                z = np.random.standard_normal(
                    (bs, seq_len, latent_dim)
                ).astype(np.float32)

                # --- Generator step ---
                with tf.GradientTape() as tape:
                    h_real = embedder(x_batch, training=True)
                    e_hat = generator(z, training=True)
                    h_hat = supervisor(e_hat, training=True)
                    x_hat = recovery(h_hat, training=True)

                    y_fake = discriminator(h_hat, training=False)

                    g_loss_u = bce(tf.ones_like(y_fake), y_fake)
                    g_loss_s = tf.reduce_mean(
                        tf.abs(h_real[:, 1:, :] - supervisor(h_real, training=True)[:, :-1, :])
                    )
                    g_loss_v = (
                        tf.reduce_mean(tf.abs(
                            tf.math.reduce_std(x_hat, axis=0)
                            - tf.math.reduce_std(x_batch, axis=0)
                        ))
                        + tf.reduce_mean(tf.abs(
                            tf.reduce_mean(x_hat, axis=0)
                            - tf.reduce_mean(x_batch, axis=0)
                        ))
                    )
                    g_loss = g_loss_u + 10.0 * g_loss_s + 100.0 * g_loss_v

                g_vars = (generator.trainable_variables
                          + supervisor.trainable_variables)
                g_grads = tape.gradient(g_loss, g_vars)
                g_optimizer.apply_gradients(zip(g_grads, g_vars))

                # --- Discriminator step ---
                with tf.GradientTape() as tape:
                    h_real = embedder(x_batch, training=False)
                    e_hat = generator(z, training=False)
                    h_hat = supervisor(e_hat, training=False)

                    y_real = discriminator(h_real, training=True)
                    y_fake = discriminator(h_hat, training=True)

                    d_loss = (
                        bce(tf.ones_like(y_real), y_real)
                        + bce(tf.zeros_like(y_fake), y_fake)
                    )

                d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(
                    zip(d_grads, discriminator.trainable_variables)
                )

                g_total += g_loss.numpy()
                d_total += d_loss.numpy()
                v_total += g_loss_v.numpy()

            # Validation moment matching loss
            n_val = len(X_val)
            z_val = np.random.standard_normal(
                (n_val, seq_len, latent_dim)
            ).astype(np.float32)
            e_val_hat = generator(z_val, training=False)
            h_val_hat = supervisor(e_val_hat, training=False)
            x_val_hat = recovery(h_val_hat, training=False)
            val_moment = (
                tf.reduce_mean(tf.abs(
                    tf.math.reduce_std(x_val_hat, axis=0)
                    - tf.math.reduce_std(X_val, axis=0)
                ))
                + tf.reduce_mean(tf.abs(
                    tf.reduce_mean(x_val_hat, axis=0)
                    - tf.reduce_mean(X_val, axis=0)
                ))
            ).numpy()

            if val_moment < best_val_moment - min_delta:
                best_val_moment = val_moment
                wait = 0
                best_weights = {
                    "generator": [w.numpy().copy() for w in generator.trainable_variables],
                    "supervisor": [w.numpy().copy() for w in supervisor.trainable_variables],
                    "embedder": [w.numpy().copy() for w in embedder.trainable_variables],
                    "recovery": [w.numpy().copy() for w in recovery.trainable_variables],
                    "discriminator": [w.numpy().copy() for w in discriminator.trainable_variables],
                }
            else:
                wait += 1

            if (epoch + 1) % 10 == 0:
                log.info(
                    f"  JOINT epoch {epoch+1}/{epochs}: "
                    f"g_loss={g_total/steps:.4f}, d_loss={d_total/steps:.4f}, "
                    f"train_moment={v_total/steps:.6f}, val_moment={val_moment:.6f} "
                    f"(best_val={best_val_moment:.6f}, wait={wait}/{patience})"
                )

            if wait >= patience:
                log.info(f"  JOINT early stop at epoch {epoch+1}, best_val_moment={best_val_moment:.6f}")
                break

        if best_weights is not None:
            for model, name in [(generator, "generator"), (supervisor, "supervisor"),
                                (embedder, "embedder"), (recovery, "recovery"),
                                (discriminator, "discriminator")]:
                for var, val in zip(model.trainable_variables, best_weights[name]):
                    var.assign(val)
