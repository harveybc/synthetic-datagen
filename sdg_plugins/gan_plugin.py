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
        "population_size": 15,        # Number of individuals per generation
        "n_generations": 20,          # Number of evolutionary iterations
        "cxpb": 0.6,                  # Crossover probability
        "mutpb": 0.3,                 # Mutation probability
        "hyperparameter_bounds": {    # Bounds for each hyperparameter
            "latent_dim": (4, 64),
            "mmd_lambda": (1e-5, 1e-2),
            "kl_beta": (1e-5, 1e-2),
            "batch_size": (16, 128),
        },
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
            raise RuntimeError("GeneratorPlugin must expose attribute 'model'.")

        # 2) Build discriminator
        self.discriminator = self._build_discriminator(config)

        # 3) Compile discriminator and adversarial model
        self.discriminator.compile(optimizer="adam", loss="binary_crossentropy")
        real_input = gen_model.input
        fake_output = self.discriminator(gen_model.output)
        self.adversarial = Model(inputs=real_input, outputs=fake_output)
        self.adversarial.compile(optimizer="adam", loss="binary_crossentropy")

        # 4) Training loop
        epochs = self.params.get("n_generations", 1)
        batch_size = self.params.get("batch_size", 32)
        for epoch in range(epochs):
            for real_batch in feeder_plugin.fetch_batch():
                # Generate noise and fake batch
                noise = generator_plugin.sample_noise(batch_size)
                fake_batch = gen_model.predict(noise)

                # Train discriminator
                self.discriminator.train_on_batch(real_batch, np.ones((len(real_batch), 1)))
                self.discriminator.train_on_batch(fake_batch, np.zeros((len(fake_batch), 1)))

                # Train generator via adversarial model
                noise = generator_plugin.sample_noise(batch_size)
                self.adversarial.train_on_batch(noise, np.ones((len(noise), 1)))

        # 5) Store trained generator
        self.trained_generator = gen_model
        return {}

    def get_trained_generator(self) -> Any:
        """
        Return the GAN-trained generator model.
        """
        return getattr(self, "trained_generator", None)

    def _build_discriminator(self, config: Dict[str, Any]) -> Any:
        """
        Build a simple discriminator model. Override for custom architectures.
        """
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Flatten, InputLayer

        input_shape = config.get("input_shape", (None,))  # adjust as needed
        model = Sequential([
            InputLayer(input_shape=input_shape),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        return model
