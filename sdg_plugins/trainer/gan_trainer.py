"""
Pure GAN Trainer plugin (no VAE encoder).

    trainer = GanTrainer()
    trainer.configure({...})
    trainer.train(train_data=["d1.csv"], save_model="model.keras")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.data_processor import prepare_training_data
from sdg_plugins.trainer.vae_gan_trainer import (
    _build_encoder,
    _build_decoder,
    _build_discriminator,
    save_model_parts,
)

log = logging.getLogger(__name__)


class GanTrainer:
    """Plugin: trains a plain GAN (generator = decoder arch)."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {}
        if config:
            self.cfg.update(config)

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    def train(
        self,
        train_data: list[str] | None = None,
        save_model: str | None = None,
    ) -> str:
        cfg = self.cfg
        paths = train_data or cfg["train_data"]
        save_path = save_model or cfg["save_model"]
        ws = cfg.get("window_size", 144)
        ld = cfg.get("latent_dim", 16)
        epochs = cfg.get("epochs", 400)
        bs = cfg.get("batch_size", 128)

        windows, initial_price = prepare_training_data(
            paths, ws, use_returns=cfg.get("use_returns", True),
        )
        log.info(f"Training windows: {windows.shape}")

        generator = _build_decoder(ws, ld, cfg)
        disc = _build_discriminator(ws, cfg)
        gen_opt = keras.optimizers.Adam(cfg.get("generator_lr", 1e-4))
        disc_opt = keras.optimizers.Adam(cfg.get("discriminator_lr", 1e-4))

        dataset = tf.data.Dataset.from_tensor_slices(
            windows.astype(np.float32)
        ).shuffle(len(windows)).batch(bs).prefetch(tf.data.AUTOTUNE)

        for epoch in range(1, epochs + 1):
            ep_d, ep_g = [], []
            for batch in dataset:
                noise = tf.random.normal((tf.shape(batch)[0], ld))
                with tf.GradientTape() as tape:
                    fake = generator(noise, training=False)
                    real_p = disc(batch, training=True)
                    fake_p = disc(fake, training=True)
                    d_loss = (
                        tf.reduce_mean(keras.losses.binary_crossentropy(
                            tf.ones_like(real_p), real_p))
                        + tf.reduce_mean(keras.losses.binary_crossentropy(
                            tf.zeros_like(fake_p), fake_p))
                    ) / 2.0
                grads = tape.gradient(d_loss, disc.trainable_variables)
                disc_opt.apply_gradients(zip(grads, disc.trainable_variables))

                noise = tf.random.normal((tf.shape(batch)[0], ld))
                with tf.GradientTape() as tape:
                    fake = generator(noise, training=True)
                    fake_p = disc(fake, training=False)
                    g_loss = tf.reduce_mean(keras.losses.binary_crossentropy(
                        tf.ones_like(fake_p), fake_p))
                grads = tape.gradient(g_loss, generator.trainable_variables)
                gen_opt.apply_gradients(zip(grads, generator.trainable_variables))

                ep_d.append(float(d_loss))
                ep_g.append(float(g_loss))

            if epoch % 10 == 0 or epoch == 1:
                log.info(f"Epoch {epoch:4d}/{epochs} │ D={np.mean(ep_d):.4f} G={np.mean(ep_g):.4f}")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        dummy_enc = _build_encoder(ws, ld, cfg)
        save_model_parts(dummy_enc, generator, save_path, initial_price)
        return save_path
