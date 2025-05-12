# optimizer/plugins/deap_optimizer.py

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


class OptimizerPlugin:
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
        "population_size": 30,        # Number of individuals per generation
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

    def __init__(self) -> None:
        """
        Initialize optimizer plugin with default parameters.
        """
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
        config: Dict[str, Any],
    ) -> Dict[str, Union[int, float]]:
        """
        Run genetic algorithm to find best hyperparameters.

        Parameters
        ----------
        feeder_plugin : Any
            Plugin with generate(n_samples, latent_dim) method.
        generator_plugin : Any
            Plugin with generate(Z) method.
        evaluator_plugin : Any
            Plugin with evaluate(synthetic_data, real_data) method.
        config : dict
            Global configuration dictionary.

        Returns
        -------
        Dict[str, Union[int, float]]
            Best hyperparameters found.
        """
        # Set random seed for reproducibility if provided
        seed = self.params.get("random_seed")
        if seed is not None:
            random.seed(seed)
            logger.info(f"Random seed set to {seed}")

        # Extract hyperparameter search space
        bounds = self.params["hyperparameter_bounds"]
        hyper_keys: List[str] = list(bounds.keys())
        low_bounds: List[float] = [bounds[k][0] for k in hyper_keys]
        up_bounds: List[float] = [bounds[k][1] for k in hyper_keys]
        int_params = {"latent_dim", "batch_size"}  # Keys that must be int

        # DEAP creator setup (avoid redefining if exists)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox configuration
        toolbox = base.Toolbox()

        # Attribute generator: float in [low, up]
        def gen_float(low: float, up: float) -> float:
            return random.uniform(low, up)

        # Register individual attributes
        for key, low, up in zip(hyper_keys, low_bounds, up_bounds):
            toolbox.register(f"attr_{key}", gen_float, low, up)

        # Individual and population registration
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.__getattribute__(f"attr_{key}") for key in hyper_keys),
            n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        def eval_individual(ind: List[float]) -> Tuple[float]:
            """
            Evaluate one individual: generate data and compute fitness (MAE).

            Parameters
            ----------
            ind : List[float]
                Genes representing hyperparameter values.

            Returns
            -------
            Tuple[float]
                Single-element tuple with fitness (lower is better).
            """
            # Map individual to dict of hyperparameters
            hp: Dict[str, Union[int, float]] = {}
            for i, key in enumerate(hyper_keys):
                val: Union[int, float] = ind[i]
                if key in int_params:
                    val = int(round(val))
                hp[key] = val

            logger.info(f"Evaluating individual: {hp}")

            # Merge with base config
            cfg = copy.deepcopy(config)
            cfg.update(hp)

            # Generate latent codes and synthetic windows
            Z = feeder_plugin.generate(
                n_samples=cfg["n_samples"], latent_dim=hp["latent_dim"]
            )
            X_syn = generator_plugin.generate(Z)

            # Evaluate synthetic data against real reference
            metrics = evaluator_plugin.evaluate(
                synthetic_data=X_syn, real_data=cfg["real_data_file"]
            )
            # Extract MAE (assumed returned key)
            fitness = metrics.get("mae", float("inf"))
            return (fitness,)

        # Register genetic operators
        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=low_bounds,
            up=up_bounds,
            eta=20.0,
            indpb=0.2,
        )
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Initialize population
        pop_size = self.params["population_size"]
        pop = toolbox.population(n=pop_size)
        logger.info(f"Initialized population of size {pop_size}")

        # Run evolutionary algorithm
        start_time = time.time()
        pop, _ = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=self.params["cxpb"],
            mutpb=self.params["mutpb"],
            ngen=self.params["n_generations"],
            verbose=True,
        )
        elapsed = time.time() - start_time
        logger.info(f"GA completed in {elapsed:.2f}s")

        # Select best individual
        best_ind = tools.selBest(pop, k=1)[0]
        best_params: Dict[str, Union[int, float]] = {}
        for i, key in enumerate(hyper_keys):
            val = best_ind[i]
            if key in int_params:
                val = int(round(val))
            best_params[key] = val

        logger.info(f"Best hyperparameters: {best_params}")
        return best_params
