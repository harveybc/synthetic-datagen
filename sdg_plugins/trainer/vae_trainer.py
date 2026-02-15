"""
Pure VAE Trainer plugin (no adversarial component).

Simpler alternative to vae_gan_trainer — useful for ablation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from app.data_processor import prepare_training_data
from sdg_plugins.trainer.vae_gan_trainer import (
    Sampling,
    _build_encoder,
    _build_decoder,
    _mmd_loss,
)

log = logging.getLogger(__name__)


class VaeTrainer:
    """Plugin: trains a plain VAE on typical_price windows."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def set_params(self, **kw):
        self.cfg.update(kw)

    def train(self) -> str:
        cfg = self.cfg
        ws = cfg["window_size"]
        ld = cfg["latent_dim"]
        epochs = cfg["epochs"]
        bs = cfg["batch_size"]

        windows, initial_price = prepare_training_data(
            cfg["train_data"], ws,
            use_returns=cfg["use_returns"],
            
        )
        log.info(f"Training windows: {windows.shape}")

        encoder = _build_encoder(ws, ld, cfg)
        decoder = _build_decoder(ws, ld, cfg)
        opt = keras.optimizers.Adam(cfg["learning_rate"])

        dataset = tf.data.Dataset.from_tensor_slices(
            windows.astype(np.float32)
        ).shuffle(len(windows)).batch(bs).prefetch(tf.data.AUTOTUNE)

        kl_max = cfg["kl_weight"]
        kl_anneal = cfg["kl_anneal_epochs"]
        best_loss = float("inf")
        patience_ctr = 0

        for epoch in range(1, epochs + 1):
            kl_w = kl_max * min(1.0, epoch / max(kl_anneal, 1))
            ep_losses = []

            for batch in dataset:
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = encoder(batch, training=True)
                    recon = decoder(z, training=True)
                    recon_loss = tf.reduce_mean(tf.square(batch - recon))
                    kl_loss = -0.5 * tf.reduce_mean(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    mmd = _mmd_loss(batch, recon)
                    loss = recon_loss + kl_w * kl_loss + cfg["mmd_lambda"] * mmd

                all_vars = encoder.trainable_variables + decoder.trainable_variables
                grads = tape.gradient(loss, all_vars)
                opt.apply_gradients(zip(grads, all_vars))
                ep_losses.append(float(loss))

            avg = np.mean(ep_losses)
            if epoch % 10 == 0 or epoch == 1:
                log.info(f"Epoch {epoch:4d}/{epochs} │ loss={avg:.6f}")

            if epoch >= cfg.get("start_from_epoch", 15):
                if avg < best_loss - cfg.get("min_delta", 1e-7):
                    best_loss = avg
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= cfg["early_patience"]:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        save_path = cfg["save_model"]
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        from sdg_plugins.trainer.vae_gan_trainer import VaeGanTrainer
        VaeGanTrainer._save(encoder, decoder, save_path, initial_price)
        return save_path
