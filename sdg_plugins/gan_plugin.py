# optimizer/plugins/gan_plugin.py

"""
Optimizer Plugin using DEAP for Synthetic Data Generator.

This plugin employs a genetic algorithm to tune key hyperparameters
for the synthetic-data generation pipeline, optimizing downstream
predictor performance.

Plugin Parameters
-----------------
- population_size (int): Number of individuals in each generation.
- n_generations (int): Number of evolutionary generations to run.
- cxpb (float): Crossover probability.
- mutpb (float): Mutation probability.
- hyperparameter_bounds (dict): Bounds for each hyperparameter to optimize.

Methods
-------
- set_params(**kwargs)
- optimize(feeder_plugin, generator_plugin, evaluator_plugin, config)
"""

import copy  # For deep-copying configuration dicts
import logging  # Standard logging module
import random  # Random number generation
import time  # Timing execution
from typing import Any, Dict, List, Tuple, Union

from deap import algorithms, base, creator, tools  # DEAP components

# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set default log level


class GANTrainerPlugin:
    """
    DEAP-based optimizer plugin for synthetic data generation.

    This plugin tunes:
      - latent_dim (int)
      - mmd_lambda (float)
      - kl_beta (float)
      - batch_size (int)

    Attributes
    ----------
    params : Dict[str, Any]
        Copy of plugin_params merged with user configuration.
    """

    #: Default optimizer configuration
    plugin_params = {
        "gan_epochs": 100,             # Number of epochs for GAN training
        "gan_batch_size": 32,          # Batch size for GAN training
        "generator_lr": 0.0002,        # Learning rate for the generator
        "discriminator_lr": 0.0002,    # Learning rate for the discriminator
        "adam_beta1": 0.5,             # Adam optimizer beta1
        # "input_sequence_length": 64, # Example: User should configure this based on their data
        # "num_features": 54,          # Example: User should configure this
        "discriminator_lstm_units": 64, # Units for the LSTM layer in discriminator
        "discriminator_dense_units": 128,# Units for the Dense layer in discriminator
         "random_seed": None,          # Optional seed for reproducibility
     }
     #: Keys included in debug output
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize optimizer plugin with default parameters.
        """
        if config is None:
            raise ValueError("Se requiere el diccionario de configuración ('config').")
        # Copia parámetros por defecto y aplica la configuración
        self.params = self.plugin_params.copy()
        self.set_params(**config)
        # Deep copy to avoid mutating the class attribute
        self.params: Dict[str, Any] = copy.deepcopy(self.plugin_params)

    def set_params(self, **kwargs: Any) -> None:
        """
        Update plugin parameters from global configuration.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments to update internal params.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Retrieve debugging information.

        Returns
        -------
        Dict[str, Any]
            Subset of params useful for debugging.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def optimize(
        self,
        feeder_plugin: Any,
        generator_plugin: Any,
        evaluator_plugin: Any,
        preprocessor_plugin: Any,
        config: Dict[str, Any],
    ) -> Dict:
        """
        Train GAN by alternating discriminator and generator updates.
        """
        import numpy as np
        from tensorflow.keras.models import Model

        # 1) Acquire generator model
        gen_model = getattr(generator_plugin, "model", None)
        if gen_model is None:
            raise RuntimeError("GANTrainerPlugin: GeneratorPlugin must expose attribute 'model'.")
        
        # Ensure the generator's output shape is defined for the discriminator
        # Expected: (batch_size, sequence_length, num_features)
        generator_output_shape = gen_model.output_shape[1:] # Exclude batch_size
        print(f"GANTrainerPlugin: Using generator output shape for discriminator input: {generator_output_shape}")

        # 2) Build discriminator
        self.discriminator = self._build_discriminator(config, input_shape=generator_output_shape)

        # 3) Compile discriminator and adversarial model
        from tensorflow.keras.optimizers import Adam
        d_optimizer = Adam(learning_rate=self.params.get("discriminator_lr"), beta_1=self.params.get("adam_beta1"))
        g_optimizer = Adam(learning_rate=self.params.get("generator_lr"), beta_1=self.params.get("adam_beta1"))

        self.discriminator.compile(optimizer=d_optimizer, loss="binary_crossentropy", metrics=['accuracy'])
        real_input = gen_model.input
        fake_output = self.discriminator(gen_model.output)
        self.adversarial = Model(inputs=real_input, outputs=fake_output)
        # For the adversarial model, we only want to train the generator
        self.discriminator.trainable = False 
        self.adversarial.compile(optimizer=g_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

        # 4) Training loop
        epochs = self.params.get("gan_epochs", 1)
        batch_size = self.params.get("gan_batch_size", 32)
        
        print(f"GANTrainerPlugin: Starting GAN training for {epochs} epochs with batch size {batch_size}.")

        for epoch in range(epochs):
            epoch_d_loss_real = []
            epoch_d_loss_fake = []
            epoch_g_loss = []
            
            for real_batch in feeder_plugin.fetch_batch():
                if real_batch.shape[0] != batch_size:
                    # If the last batch from feeder is smaller, adjust noise and fake batch size
                    current_batch_size = real_batch.shape[0]
                    if current_batch_size == 0: continue
                else:
                    current_batch_size = batch_size

                 # Generate noise and fake batch
                noise = generator_plugin.sample_noise(current_batch_size) # Use current_batch_size
                fake_batch = gen_model.predict(noise)

                # Create labels
                real_labels = np.ones((current_batch_size, 1))
                fake_labels = np.zeros((current_batch_size, 1))

                # Train discriminator
                self.discriminator.trainable = True # Ensure discriminator is trainable
                d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
                epoch_d_loss_real.append(d_loss_real[0]) # loss value
                epoch_d_loss_fake.append(d_loss_fake[0]) # loss value

                 # Train generator via adversarial model
                # For adversarial training, discriminator weights are frozen (done during compile)
                self.discriminator.trainable = False 
                noise_for_g = generator_plugin.sample_noise(current_batch_size) # Use current_batch_size
                g_loss = self.adversarial.train_on_batch(noise_for_g, real_labels) # Generator tries to make discriminator output 1 (real)
                epoch_g_loss.append(g_loss[0]) # loss value
            
            avg_d_loss_real = np.mean(epoch_d_loss_real) if epoch_d_loss_real else 0
            avg_d_loss_fake = np.mean(epoch_d_loss_fake) if epoch_d_loss_fake else 0
            avg_g_loss = np.mean(epoch_g_loss) if epoch_g_loss else 0
            print(f"Epoch {epoch+1}/{epochs} -> D_Loss_Real: {avg_d_loss_real:.4f}, D_Loss_Fake: {avg_d_loss_fake:.4f}, G_Loss: {avg_g_loss:.4f}")

        # 5) Store trained generator
        self.trained_generator = gen_model
        print("GANTrainerPlugin: GAN training completed.")
        return {}

    def get_trained_generator(self) -> Any:
        """
        Return the GAN-trained generator model.
        """
        return getattr(self, "trained_generator", None)

    def _build_discriminator(self, config: Dict[str, Any], input_shape: Tuple) -> Any:
        """
        Build a simple discriminator model. Override for custom architectures.
        Now uses an LSTM layer suitable for sequence data.
        """
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Input

        # input_shape should be (sequence_length, num_features), e.g., (64, 54)
        # This is now taken from generator's output shape.
        lstm_units = self.params.get("discriminator_lstm_units", 64)
        dense_units = self.params.get("discriminator_dense_units", 128)
        
        print(f"GANTrainerPlugin: Building discriminator with input shape {input_shape}, LSTM units: {lstm_units}, Dense units: {dense_units}")

        model = Sequential([
            Input(shape=input_shape), # Use Input layer for explicit shape
            LSTM(lstm_units), # LSTM layer to process sequences
            Dense(dense_units, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        return model
