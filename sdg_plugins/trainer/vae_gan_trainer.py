"""
VAE-GAN Trainer plugin.

Programmatic API:

    from sdg_plugins.trainer.vae_gan_trainer import VaeGanTrainer
    trainer = VaeGanTrainer()
    trainer.configure({...})
    model_path = trainer.train(
        train_data=["d1.csv", "d2.csv"],
        save_model="model.keras",
    )

Architecture ported & simplified from timeseries-gan (SC-VAE-GAN).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from app.data_processor import prepare_training_data

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Sampling layer
# ═══════════════════════════════════════════════════════════════════════════

class Sampling(layers.Layer):
    """Reparameterisation trick: z = mu + sigma * epsilon."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ═══════════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_encoder(window_size: int, latent_dim: int, cfg: Dict) -> keras.Model:
    act = cfg.get("activation", "tanh")
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-6))
    n_layers = cfg.get("intermediate_layers", 2)
    size = cfg.get("initial_layer_size", 48)
    divisor = cfg.get("layer_size_divisor", 2)

    inp = keras.Input(shape=(window_size,), name="encoder_input")
    x = inp
    for i in range(n_layers):
        units = max(size // (divisor ** i), latent_dim)
        x = layers.Dense(units, activation=act, kernel_regularizer=l2,
                          name=f"enc_dense_{i}")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    return keras.Model(inp, [z_mean, z_log_var, z], name="encoder")


def _build_decoder(window_size: int, latent_dim: int, cfg: Dict) -> keras.Model:
    act = cfg.get("activation", "tanh")
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-6))
    n_layers = cfg.get("intermediate_layers", 2)
    size = cfg.get("initial_layer_size", 48)
    divisor = cfg.get("layer_size_divisor", 2)

    inp = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = inp
    for i in range(n_layers - 1, -1, -1):
        units = max(size // (divisor ** i), latent_dim)
        x = layers.Dense(units, activation=act, kernel_regularizer=l2,
                          name=f"dec_dense_{i}")(x)
    out = layers.Dense(window_size, activation="linear", name="decoder_output")(x)
    return keras.Model(inp, out, name="decoder")


def _build_discriminator(window_size: int, cfg: Dict) -> keras.Model:
    inp = keras.Input(shape=(window_size,), name="disc_input")
    x = inp
    for i, units in enumerate(cfg.get("disc_layers", [64, 32])):
        x = layers.Dense(units, name=f"disc_dense_{i}")(x)
        x = layers.LeakyReLU(0.2, name=f"disc_lrelu_{i}")(x)
        x = layers.Dropout(cfg.get("disc_dropout", 0.3), name=f"disc_drop_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="disc_out")(x)
    return keras.Model(inp, out, name="discriminator")


# ═══════════════════════════════════════════════════════════════════════════
# MMD loss
# ═══════════════════════════════════════════════════════════════════════════

def _mmd_loss(real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """Gaussian-kernel MMD² with median heuristic bandwidth."""
    dists = tf.reduce_sum(tf.square(real[:, None] - real[None, :]), axis=-1)
    gamma = 1.0 / (tf.reduce_mean(dists) + 1e-8)
    xx = tf.exp(-gamma * tf.reduce_sum(tf.square(real[:, None] - real[None, :]), -1))
    yy = tf.exp(-gamma * tf.reduce_sum(tf.square(fake[:, None] - fake[None, :]), -1))
    xy = tf.exp(-gamma * tf.reduce_sum(tf.square(real[:, None] - fake[None, :]), -1))
    return tf.reduce_mean(xx) + tf.reduce_mean(yy) - 2.0 * tf.reduce_mean(xy)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer plugin
# ═══════════════════════════════════════════════════════════════════════════

class VaeGanTrainer:
    """
    Plugin: trains a VAE-GAN on typical_price windows.

    Discoverable via entry_points as ``sdg.trainer → vae_gan_trainer``.
    """

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {}
        if config:
            self.cfg.update(config)

    # ── Plugin interface ────────────────────────────────────────────────

    def configure(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        """Alias for configure (Harvey's convention)."""
        self.cfg.update(kw)

    def train(
        self,
        train_data: list[str] | None = None,
        save_model: str | None = None,
    ) -> str:
        """
        Train the VAE-GAN model.

        Args:
            train_data: List of CSV paths. Falls back to self.cfg["train_data"].
            save_model: Output model path. Falls back to self.cfg["save_model"].

        Returns:
            Path to the saved model.
        """
        cfg = self.cfg
        paths = train_data or cfg["train_data"]
        save_path = save_model or cfg["save_model"]

        ws = cfg.get("window_size", 144)
        ld = cfg.get("latent_dim", 16)
        epochs = cfg.get("epochs", 400)
        bs = cfg.get("batch_size", 128)

        # Data
        log.info("Loading training data …")
        windows, initial_price = prepare_training_data(
            paths, ws, use_returns=cfg.get("use_returns", True),
        )
        log.info(f"Training windows: {windows.shape}")

        # Models
        encoder = _build_encoder(ws, ld, cfg)
        decoder = _build_decoder(ws, ld, cfg)
        disc = _build_discriminator(ws, cfg)

        # Optimisers
        vae_opt = keras.optimizers.Adam(cfg.get("learning_rate", 1e-3))
        disc_opt = keras.optimizers.Adam(cfg.get("discriminator_lr", 1e-4))
        gen_opt = keras.optimizers.Adam(cfg.get("generator_lr", 1e-4))

        # KL annealing
        kl_max = cfg.get("kl_weight", 1e-3)
        kl_anneal = cfg.get("kl_anneal_epochs", 40)

        dataset = tf.data.Dataset.from_tensor_slices(
            windows.astype(np.float32)
        ).shuffle(len(windows)).batch(bs).prefetch(tf.data.AUTOTUNE)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            kl_w = kl_max * min(1.0, epoch / max(kl_anneal, 1))
            ep_recon, ep_kl, ep_mmd, ep_d, ep_g = [], [], [], [], []

            for batch in dataset:
                # ── VAE step ──
                vae_vars = encoder.trainable_variables + decoder.trainable_variables
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = encoder(batch, training=True)
                    recon = decoder(z, training=True)
                    recon_loss = tf.reduce_mean(tf.square(batch - recon))
                    kl_loss = -0.5 * tf.reduce_mean(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    mmd = _mmd_loss(batch, recon)
                    vae_loss = recon_loss + kl_w * kl_loss + cfg.get("mmd_lambda", 1e-2) * mmd
                grads = tape.gradient(vae_loss, vae_vars)
                vae_opt.apply_gradients(zip(grads, vae_vars))

                # ── Discriminator step ──
                noise_z = tf.random.normal((tf.shape(batch)[0], ld))
                fake = decoder(noise_z, training=False)
                with tf.GradientTape() as tape:
                    real_pred = disc(batch, training=True)
                    fake_pred = disc(fake, training=True)
                    d_loss = (
                        tf.reduce_mean(keras.losses.binary_crossentropy(
                            tf.ones_like(real_pred), real_pred))
                        + tf.reduce_mean(keras.losses.binary_crossentropy(
                            tf.zeros_like(fake_pred), fake_pred))
                    ) / 2.0
                grads = tape.gradient(d_loss, disc.trainable_variables)
                disc_opt.apply_gradients(zip(grads, disc.trainable_variables))

                # ── Generator adversarial step ──
                noise_z = tf.random.normal((tf.shape(batch)[0], ld))
                with tf.GradientTape() as tape:
                    fake = decoder(noise_z, training=True)
                    fake_pred = disc(fake, training=False)
                    g_loss = tf.reduce_mean(keras.losses.binary_crossentropy(
                        tf.ones_like(fake_pred), fake_pred))
                grads = tape.gradient(g_loss, decoder.trainable_variables)
                gen_opt.apply_gradients(zip(grads, decoder.trainable_variables))

                ep_recon.append(float(recon_loss))
                ep_kl.append(float(kl_loss))
                ep_mmd.append(float(mmd))
                ep_d.append(float(d_loss))
                ep_g.append(float(g_loss))

            avg_recon = np.mean(ep_recon)
            avg_kl = np.mean(ep_kl)
            avg_mmd = np.mean(ep_mmd)
            total = avg_recon + kl_w * avg_kl + cfg.get("mmd_lambda", 1e-2) * avg_mmd

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:4d}/{epochs} │ "
                    f"recon={avg_recon:.6f} kl={avg_kl:.6f} mmd={avg_mmd:.6f} "
                    f"D={np.mean(ep_d):.4f} G={np.mean(ep_g):.4f} total={total:.6f}"
                )

            if epoch >= cfg.get("start_from_epoch", 15):
                if total < best_loss - cfg.get("min_delta", 1e-7):
                    best_loss = total
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= cfg.get("early_patience", 120):
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_model_parts(encoder, decoder, save_path, initial_price)
        return save_path


# ═══════════════════════════════════════════════════════════════════════════
# Shared persistence (used by all trainers)
# ═══════════════════════════════════════════════════════════════════════════

def save_model_parts(
    encoder: keras.Model,
    decoder: keras.Model,
    path: str,
    initial_price: float,
) -> None:
    """Save encoder + decoder + metadata as .keras + .parts/ directory."""
    parts_dir = path + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    encoder.save(os.path.join(parts_dir, "encoder.keras"))
    decoder.save(os.path.join(parts_dir, "decoder.keras"))

    meta = {
        "initial_price": initial_price,
        "latent_dim": int(decoder.input_shape[-1]),
        "window_size": int(decoder.output_shape[-1]),
    }
    with open(os.path.join(parts_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    decoder.save(path)
    log.info(f"Saved model parts → {parts_dir}/ and decoder → {path}")
