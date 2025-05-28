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

import numpy as np # ADD THIS IMPORT
import pandas as pd  # Ensure pandas is imported
from datetime import datetime, timedelta  # For datetime generation
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
        "population_size": 15,        # Number of individuals per generation
        "n_generations": 20,          # Number of evolutionary iterations
        "cxpb": 0.6,                  # Crossover probability
        "mutpb": 0.3,                 # Mutation probability
        "hyperparameter_bounds": {    # Bounds for each hyperparameter
            "latent_dim": (4, 64), # For FeederPlugin
            # "mmd_lambda": (1e-5, 1e-2), # Example if you tune GAN params
            # "kl_beta": (1e-5, 1e-2),  # Example if you tune GAN params
            "batch_size": (16, 128), # Example, if used by a plugin being tuned
            # Add other relevant hyperparameters for Feeder or Generator if they are tuned
            # e.g., "generator_decoder_input_window_size": (60, 200),
        },
        "optimizer_n_samples_per_eval": 100, # Number of samples to generate per evaluation
        "optimizer_start_datetime": None, # Optional: "YYYY-MM-DD HH:MM:SS" for eval consistency
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
        
        self.params = copy.deepcopy(self.plugin_params) # Deep copy defaults
        self.set_params(**config) # Apply global config overrides
        
        # Store the initial global config separately if needed by helpers
        # self.global_config_snapshot = config.copy() # If _get_config_param needs the original global config

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
        config: Dict[str, Any], # This is the main, merged config from main.py
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
        # Store plugins and config for access by eval_individual (if it becomes a method)
        # or for creating copies.
        self.feeder_ref = feeder_plugin
        self.generator_ref = generator_plugin
        self.evaluator_ref = evaluator_plugin
        self.global_config = config # Store the comprehensive config

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
        
        # Define which hyperparameters must be integers
        # This should align with the keys in "hyperparameter_bounds"
        int_params = {key for key in hyper_keys if isinstance(bounds[key][0], int) and isinstance(bounds[key][1], int)}
        # Or explicitly:
        # int_params = {"latent_dim", "batch_size"} # Add other integer params you tune

        # DEAP creator setup (avoid redefining if exists)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox configuration
        toolbox = base.Toolbox()

        # Attribute generator: float or int based on bounds
        for key, low, up in zip(hyper_keys, low_bounds, up_bounds):
            if key in int_params:
                 toolbox.register(f"attr_{key}", random.randint, int(low), int(up))
            else:
                toolbox.register(f"attr_{key}", random.uniform, low, up)


        # Individual and population registration
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.__getattribute__(f"attr_{key}") for key in hyper_keys),
            n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function (nested function, captures variables from optimize's scope)
        def eval_individual(ind: List[Union[float, int]]) -> Tuple[float]:
            """
            Evaluate one individual: generate data and compute fitness.

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
            hp_for_eval: Dict[str, Union[int, float]] = {}
            for i, key in enumerate(hyper_keys):
                val = ind[i]
                # DEAP might provide floats even for int attributes if not careful with registration
                if key in int_params:
                    hp_for_eval[key] = int(round(val))
                else:
                    hp_for_eval[key] = val
            
            logger.info(f"Evaluating individual with hyperparameters: {hp_for_eval}")

            # Create a new configuration for this specific evaluation run
            # Start with the global config and override with current hyperparameters
            current_eval_config = copy.deepcopy(self.global_config)
            current_eval_config.update(hp_for_eval)

            # Create temporary plugin instances for this evaluation to avoid side effects
            # This assumes plugins can be initialized with a config dict.
            try:
                temp_feeder = type(self.feeder_ref)(current_eval_config)
                temp_feeder.set_params(**current_eval_config)

                temp_generator = type(self.generator_ref)(current_eval_config)
                temp_generator.set_params(**current_eval_config)
                
                temp_evaluator = type(self.evaluator_ref)(current_eval_config)
                temp_evaluator.set_params(**current_eval_config) # If evaluator also has tunable params or needs updated config
            except Exception as e_init:
                logger.error(f"Failed to initialize temporary plugins for evaluation: {e_init}")
                return (float('inf'),) # Return a very bad fitness

            # Determine the number of ticks for this evaluation run
            # Use self.params for optimizer-specific settings, self.global_config for general ones
            num_ticks_for_evaluation = self.params.get('optimizer_n_samples_per_eval', 100)
            if num_ticks_for_evaluation <= 0:
                logger.error("'optimizer_n_samples_per_eval' must be positive.")
                return (float('inf'),)

            # Generate target_datetimes for the FeederPlugin
            # _generate_datetimes_for_opt needs access to the OptimizerPlugin instance (self)
            target_datetimes_for_eval = self._generate_datetimes_for_opt(num_ticks_for_evaluation)
            if target_datetimes_for_eval.empty:
                logger.error("Failed to generate target_datetimes for evaluation.")
                return (float('inf'),)

            # Call FeederPlugin.generate
            try:
                feeder_outputs_sequence = temp_feeder.generate(
                    n_ticks_to_generate=num_ticks_for_evaluation,
                    target_datetimes=target_datetimes_for_eval
                )
            except Exception as e_feeder:
                logger.error(f"FeederPlugin.generate failed during optimization eval: {e_feeder}")
                return (float('inf'),)

            # Prepare initial window for the generator
            # Use params from the temporary generator instance
            decoder_input_window_size = temp_generator.params.get("decoder_input_window_size")
            gen_full_feature_names = temp_generator.params.get("full_feature_names_ordered", [])
            
            if not gen_full_feature_names:
                logger.error("Generator's 'full_feature_names_ordered' is empty in current_eval_config.")
                return (float('inf'),)
            num_all_features_gen = len(gen_full_feature_names)
            
            if decoder_input_window_size is None or decoder_input_window_size <= 0:
                logger.error(f"Invalid 'decoder_input_window_size': {decoder_input_window_size}")
                return (float('inf'),)

            initial_window_for_generator = np.zeros((decoder_input_window_size, num_all_features_gen), dtype=np.float32)

            # Call GeneratorPlugin.generate
            try:
                generated_full_sequence_batch = temp_generator.generate(
                    feeder_outputs_sequence=feeder_outputs_sequence,
                    sequence_length_T=num_ticks_for_evaluation,
                    initial_full_feature_window=initial_window_for_generator
                )
                # Assuming batch size is 1 for generation in optimizer
                synthetic_data_np = generated_full_sequence_batch[0] 
            except Exception as e_generator:
                logger.error(f"GeneratorPlugin.generate failed during optimization eval: {e_generator}")
                return (float('inf'),)

            # Evaluate the generated synthetic_data
            try:
                # This part is highly dependent on your EvaluatorPlugin's `evaluate` method.
                # It needs to handle how it gets the 'real_data_processed'.
                # For now, we pass the current_eval_config which might contain 'real_data_file'.
                # The EvaluatorPlugin must be robust enough to load/process this.
                # A better approach would be to pre-load and pre-process a fixed evaluation
                # dataset at the start of the `optimize` method and pass the NumPy array here.
                
                # Assuming evaluator can get feature names from synthetic_data_np if not provided,
                # or uses feature names from its own config.
                metrics = temp_evaluator.evaluate(
                    synthetic_data=synthetic_data_np,
                    # real_data_processed=... # This is the tricky part.
                                                # Evaluator needs a consistent real data segment.
                                                # If it loads from file path in current_eval_config,
                                                # ensure that file is appropriate and preprocessed.
                    feature_names=gen_full_feature_names, # Or a subset if evaluator expects that
                    config=current_eval_config
                )
                # Extract a fitness score. Lower is better for FitnessMin.
                # Example: using a specific MMD score or a combined quality score.
                # This key 'mae' was from the original snippet, adjust if your metrics are different.
                fitness_score = metrics.get("mae", metrics.get("avg_mmd_rbf", float('inf'))) 
                if fitness_score is None or np.isnan(fitness_score) or not isinstance(fitness_score, (int, float)):
                    logger.warning(f"Invalid fitness score from metrics: {fitness_score}. Defaulting to inf.")
                    fitness_score = float('inf')
                
                logger.info(f"Individual {hp_for_eval} evaluated. Fitness: {fitness_score}")

            except Exception as e_evaluator:
                logger.error(f"EvaluatorPlugin.evaluate failed during optimization eval: {e_evaluator}")
                fitness_score = float('inf')

            return (fitness_score,) # DEAP expects a tuple

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
        pop, logbook = algorithms.eaSimple( # Store logbook
            pop,
            toolbox,
            cxpb=self.params["cxpb"],
            mutpb=self.params["mutpb"],
            ngen=self.params["n_generations"],
            verbose=True,
        )
        elapsed = time.time() - start_time
        logger.info(f"GA completed in {elapsed:.2f}s")
        if logbook: # Print logbook if available
            logger.info(f"Logbook: {logbook}")

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

    def _get_config_param(self, key: str, default: Any = None) -> Any:
        """Helper to get a parameter from the optimizer's main config or its own params."""
        # Prioritize global_config if available and key exists, else optimizer's params
        if hasattr(self, 'global_config') and self.global_config is not None and key in self.global_config:
            return self.global_config.get(key, default)
        return self.params.get(key, default) # Fallback to optimizer's own params

    def _generate_datetimes_for_opt(self, num_ticks: int) -> pd.Series:
        """
        Generates a sequence of datetimes for optimizer evaluation.
        """
        # Use optimizer-specific start datetime if provided, else general start_datetime
        start_dt_str = self.params.get("optimizer_start_datetime") # From optimizer's own params
        if not start_dt_str:
            start_dt_str = self._get_config_param("start_datetime", datetime.now().strftime('%Y-%m-%d %H:%M:%S')) # From global or optimizer params
        
        periodicity = self._get_config_param("dataset_periodicity", "1h") # From global or optimizer params

        try:
            current_dt = pd.to_datetime(start_dt_str)
        except Exception as e:
            logger.warning(f"Could not parse optimizer_start_datetime '{start_dt_str}'. Defaulting. Error: {e}")
            current_dt = pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        delta = timedelta(hours=1) # Default
        if 'h' in periodicity.lower():
            try: delta = timedelta(hours=int(periodicity.lower().replace('h', '')))
            except ValueError: logger.warning(f"Invalid hour value in periodicity: {periodicity}")
        elif 'min' in periodicity.lower() or 'm' in periodicity.lower() or 't' in periodicity.lower():
            try: delta = timedelta(minutes=int(periodicity.lower().replace('min', '').replace('m', '').replace('t', '')))
            except ValueError: logger.warning(f"Invalid minute value in periodicity: {periodicity}")
        elif 'd' in periodicity.lower():
            try: delta = timedelta(days=int(periodicity.lower().replace('d', '')))
            except ValueError: logger.warning(f"Invalid day value in periodicity: {periodicity}")
        else:
            logger.warning(f"Unhandled periodicity '{periodicity}' for optimizer datetime generation. Defaulting to 1 hour.")
            
        datetimes = [current_dt + i * delta for i in range(num_ticks)]
        return pd.Series(datetimes)

    # The old eval_individual function (if it was a class method) is now replaced by the nested one.
    # If you had other helper methods for eval_individual, ensure they are accessible or integrated.
