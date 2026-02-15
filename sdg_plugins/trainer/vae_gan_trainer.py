"""
VAE-GAN Trainer plugin.

Trains a VAE with adversarial (GAN) refinement on typical_price log-returns.
Architecture ported & simplified from timeseries-gan (SC-VAE-GAN).

Supports conditional mode: temporal features (hour, day-of-week, month)
are injected into encoder, decoder, and discriminator so the model learns
season-aware generation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

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

def _build_encoder(window_size: int, latent_dim: int, cfg: Dict,
                   n_temporal: int = 0) -> keras.Model:
    act = cfg["activation"]
    l2 = regularizers.l2(cfg["l2_reg"])
    n_layers = cfg["intermediate_layers"]
    size = cfg["initial_layer_size"]
    divisor = cfg["layer_size_divisor"]

    inp = keras.Input(shape=(window_size,), name="encoder_input")
    inputs = [inp]

    x = inp
    # First dense layer
    units0 = max(size, latent_dim)
    x = layers.Dense(units0, activation=act, kernel_regularizer=l2,
                     name="enc_dense_0")(x)

    # Concatenate temporal features after first layer
    if n_temporal > 0:
        temp_inp = keras.Input(shape=(n_temporal,), name="encoder_temporal_input")
        inputs.append(temp_inp)
        x = layers.Concatenate(name="enc_cat_temporal")([x, temp_inp])

    for i in range(1, n_layers):
        units = max(size // (divisor ** i), latent_dim)
        x = layers.Dense(units, activation=act, kernel_regularizer=l2,
                         name=f"enc_dense_{i}")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def _build_decoder(window_size: int, latent_dim: int, cfg: Dict,
                   n_temporal: int = 0) -> keras.Model:
    act = cfg["activation"]
    l2 = regularizers.l2(cfg["l2_reg"])
    n_layers = cfg["intermediate_layers"]
    size = cfg["initial_layer_size"]
    divisor = cfg["layer_size_divisor"]

    inp = keras.Input(shape=(latent_dim,), name="decoder_input")
    inputs = [inp]

    if n_temporal > 0:
        temp_inp = keras.Input(shape=(n_temporal,), name="decoder_temporal_input")
        inputs.append(temp_inp)
        x = layers.Concatenate(name="dec_cat_temporal")([inp, temp_inp])
    else:
        x = inp

    for i in range(n_layers - 1, -1, -1):
        units = max(size // (divisor ** i), latent_dim)
        x = layers.Dense(units, activation=act, kernel_regularizer=l2,
                         name=f"dec_dense_{i}")(x)
    out = layers.Dense(window_size, activation="linear", name="decoder_output")(x)
    return keras.Model(inputs, out, name="decoder")


def _build_discriminator(window_size: int, cfg: Dict,
                         n_temporal: int = 0) -> keras.Model:
    """Simple MLP discriminator operating on raw windows."""
    inp = keras.Input(shape=(window_size,), name="disc_input")
    inputs = [inp]

    if n_temporal > 0:
        temp_inp = keras.Input(shape=(n_temporal,), name="disc_temporal_input")
        inputs.append(temp_inp)
        x = layers.Concatenate(name="disc_cat_temporal")([inp, temp_inp])
    else:
        x = inp

    for i, units in enumerate(cfg.get("disc_layers", [64, 32])):
        x = layers.Dense(units, name=f"disc_dense_{i}")(x)
        x = layers.LeakyReLU(0.2, name=f"disc_lrelu_{i}")(x)
        x = layers.Dropout(cfg.get("disc_dropout", 0.3), name=f"disc_drop_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="disc_out")(x)
    return keras.Model(inputs, out, name="discriminator")


# ═══════════════════════════════════════════════════════════════════════════
# MMD loss
# ═══════════════════════════════════════════════════════════════════════════

def _mmd_loss(real: tf.Tensor, fake: tf.Tensor, gamma: float | None = None) -> tf.Tensor:
    """Gaussian-kernel MMD² between two batches."""
    if gamma is None:
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
    """Plugin: trains a VAE-GAN on typical_price windows."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = config or {}

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw):
        self.cfg.update(kw)

    # ── public ──────────────────────────────────────────────────────────

    def train(self, **kwargs) -> str:
        """Run full training.  Returns path to saved model."""
        if kwargs:
            self.cfg.update(kwargs)
        cfg = self.cfg
        ws = cfg["window_size"]
        ld = cfg["latent_dim"]
        epochs = cfg["epochs"]
        bs = cfg["batch_size"]
        conditional = cfg.get("conditional", False)
        n_temporal = cfg.get("n_temporal", 6) if conditional else 0

        # Data
        log.info("Loading training data …")
        if conditional:
            windows, temporal_features, initial_price = prepare_training_data(
                cfg["train_data"], ws,
                use_returns=cfg["use_returns"],
                conditional=True,
            )
            log.info(f"Training windows: {windows.shape}, temporal: {temporal_features.shape}")
        else:
            windows, initial_price = prepare_training_data(
                cfg["train_data"], ws,
                use_returns=cfg["use_returns"],
                conditional=False,
            )
            temporal_features = None
            log.info(f"Training windows: {windows.shape}")

        # Models
        encoder = _build_encoder(ws, ld, cfg, n_temporal=n_temporal)
        decoder = _build_decoder(ws, ld, cfg, n_temporal=n_temporal)
        disc = _build_discriminator(ws, cfg, n_temporal=n_temporal)

        # Optimisers
        vae_opt = keras.optimizers.Adam(cfg["learning_rate"])
        disc_opt = keras.optimizers.Adam(cfg.get("discriminator_lr", 1e-4))
        gen_opt = keras.optimizers.Adam(cfg.get("generator_lr", 1e-4))

        # KL annealing
        kl_max = cfg["kl_weight"]
        kl_anneal = cfg["kl_anneal_epochs"]

        # Training dataset
        if conditional:
            dataset = tf.data.Dataset.from_tensor_slices((
                windows.astype(np.float32),
                temporal_features.astype(np.float32),
            )).shuffle(len(windows)).batch(bs).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                windows.astype(np.float32)
            ).shuffle(len(windows)).batch(bs).prefetch(tf.data.AUTOTUNE)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            kl_w = kl_max * min(1.0, epoch / max(kl_anneal, 1))

            ep_recon, ep_kl, ep_mmd, ep_d, ep_g = [], [], [], [], []

            for batch_data in dataset:
                if conditional:
                    batch, batch_temp = batch_data
                    enc_in = [batch, batch_temp]
                    disc_real_in = [batch, batch_temp]
                else:
                    batch = batch_data
                    enc_in = batch
                    disc_real_in = batch

                # VAE step
                vae_vars = encoder.trainable_variables + decoder.trainable_variables
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = encoder(enc_in, training=True)
                    dec_in = [z, batch_temp] if conditional else z
                    recon = decoder(dec_in, training=True)
                    recon_loss = tf.reduce_mean(tf.square(batch - recon))
                    kl_loss = -0.5 * tf.reduce_mean(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    mmd = _mmd_loss(batch, recon)
                    vae_loss = recon_loss + kl_w * kl_loss + cfg["mmd_lambda"] * mmd
                grads = tape.gradient(vae_loss, vae_vars)
                vae_opt.apply_gradients(zip(grads, vae_vars))

                # Discriminator step
                noise_z = tf.random.normal((tf.shape(batch)[0], ld))
                dec_noise_in = [noise_z, batch_temp] if conditional else noise_z
                fake = decoder(dec_noise_in, training=False)
                disc_fake_in = [fake, batch_temp] if conditional else fake

                with tf.GradientTape() as tape:
                    real_pred = disc(disc_real_in, training=True)
                    fake_pred = disc(disc_fake_in, training=True)
                    d_loss = (
                        tf.reduce_mean(keras.losses.binary_crossentropy(
                            tf.ones_like(real_pred), real_pred))
                        + tf.reduce_mean(keras.losses.binary_crossentropy(
                            tf.zeros_like(fake_pred), fake_pred))
                    ) / 2.0
                grads = tape.gradient(d_loss, disc.trainable_variables)
                disc_opt.apply_gradients(zip(grads, disc.trainable_variables))

                # Generator adversarial step
                noise_z = tf.random.normal((tf.shape(batch)[0], ld))
                with tf.GradientTape() as tape:
                    dec_noise_in = [noise_z, batch_temp] if conditional else noise_z
                    fake = decoder(dec_noise_in, training=True)
                    disc_fake_in = [fake, batch_temp] if conditional else fake
                    fake_pred = disc(disc_fake_in, training=False)
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
            avg_d = np.mean(ep_d)
            avg_g = np.mean(ep_g)
            total = avg_recon + kl_w * avg_kl + cfg["mmd_lambda"] * avg_mmd

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:4d}/{epochs} │ "
                    f"recon={avg_recon:.6f} kl={avg_kl:.6f} mmd={avg_mmd:.6f} "
                    f"D={avg_d:.4f} G={avg_g:.4f} total={total:.6f}"
                )

            # Early stopping
            if epoch >= cfg.get("start_from_epoch", 15):
                if total < best_loss - cfg.get("min_delta", 1e-7):
                    best_loss = total
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= cfg["early_patience"]:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        # Save
        save_path = cfg["save_model"]
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        self._save(encoder, decoder, save_path, initial_price,
                   conditional=conditional, n_temporal=n_temporal)
        return save_path

    # ── persistence ─────────────────────────────────────────────────────

    @staticmethod
    def _save(encoder, decoder, path, initial_price,
              conditional=False, n_temporal=0):
        """Save encoder + decoder + metadata as .parts/ archive."""
        combined_dir = path + ".parts"
        os.makedirs(combined_dir, exist_ok=True)
        encoder.save(os.path.join(combined_dir, "encoder.keras"))
        decoder.save(os.path.join(combined_dir, "decoder.keras"))

        # Determine latent_dim and window_size from model shapes
        dec_out = decoder.output_shape
        window_size = int(dec_out[-1])

        # For conditional decoder, first input is latent, second is temporal
        if conditional:
            latent_dim = int(decoder.input_shape[0][-1])
        else:
            latent_dim = int(decoder.input_shape[-1])

        meta = {
            "initial_price": initial_price,
            "latent_dim": latent_dim,
            "window_size": window_size,
            "conditional": conditional,
            "n_temporal": n_temporal,
        }
        with open(os.path.join(combined_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Also save decoder standalone for quick non-conditional use
        decoder.save(path)
        log.info(f"Saved model parts → {combined_dir}/ and decoder → {path}")
