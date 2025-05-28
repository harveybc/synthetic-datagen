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

    def _get_config_param(self, key: str, default: Any = None) -> Any:
        """Helper to get a parameter from the optimizer's stored config."""
        # Assuming self.config is populated with the main configuration
        # during __init__ or set_params or when optimize is called.
        if hasattr(self, "config") and isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def _generate_datetimes_for_opt(self, num_ticks: int) -> pd.Series:
        """
        Generates a sequence of datetimes for optimizer evaluation.
        This is a simplified version. For more complex scenarios (e.g., skipping weekends),
        you might need to adapt logic from main.py's generate_datetime_column or
        use pre-defined evaluation datetime sequences.
        """
        start_dt_str = self._get_config_param("optimizer_start_datetime")
        if not start_dt_str:  # Fallback to a default if not specified for optimizer
            start_dt_str = self._get_config_param(
                "start_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        periodicity = self._get_config_param("dataset_periodicity", "1h")

        try:
            current_dt = pd.to_datetime(start_dt_str)
        except Exception as e:
            print(
                f"OptimizerPlugin: Warning - Could not parse optimizer_start_datetime '{start_dt_str}'. Defaulting. Error: {e}"
            )
            current_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Simplified timedelta conversion
        if "h" in periodicity:
            delta = timedelta(hours=int(periodicity.replace("h", "")))
        elif "min" in periodicity or "m" in periodicity or "T" in periodicity:
            delta = timedelta(
                minutes=int(periodicity.replace("min", "").replace("m", "").replace("T", ""))
            )
        elif "D" in periodicity:
            delta = timedelta(days=int(periodicity.replace("D", "")))
        else:
            print(
                f"OptimizerPlugin: Warning - Unhandled periodicity '{periodicity}' for optimizer datetime generation. Defaulting to 1 hour."
            )
            delta = timedelta(hours=1)

        datetimes = [current_dt + i * delta for i in range(num_ticks)]
        return pd.Series(datetimes)

    def eval_individual(self, individual_hyperparams):
        """
        Evaluates an individual set of hyperparameters.
        This is a common function name for evolutionary algorithms (e.g., used with DEAP).
        """
        # 1. Apply individual_hyperparams to copies of feeder, generator, evaluator plugins
        #    or re-initialize them with these hyperparameters.
        #    This part is crucial and depends on your OptimizerPlugin's design.
        #    Example:
        #    temp_feeder_config = self.feeder.params.copy()
        #    temp_feeder_config.update(individual_hyperparams relevant to feeder)
        #    current_feeder_eval = type(self.feeder)(temp_feeder_config) # Or use set_params on a copy
        #    current_feeder_eval.set_params(**temp_feeder_config)

        #    temp_generator_config = self.generator.params.copy()
        #    temp_generator_config.update(individual_hyperparams relevant to generator)
        #    current_generator_eval = type(self.generator)(temp_generator_config)
        #    current_generator_eval.set_params(**temp_generator_config)

        # For this example, let's assume self.feeder and self.generator are
        # already configured with the current individual's hyperparameters.

        # 2. Determine the number of ticks for this evaluation run
        #    This might come from the optimizer's config or be fixed.
        num_ticks_for_evaluation = self._get_config_param(
            "optimizer_n_samples_per_eval", 100
        )  # Example value

        # 3. Generate target_datetimes for the FeederPlugin
        target_datetimes_for_eval = self._generate_datetimes_for_opt(num_ticks_for_evaluation)

        # 4. Call FeederPlugin.generate with the correct arguments
        #    The variable that was 'Z' should now store the sequence of feeder outputs.
        feeder_outputs_sequence = self.feeder.generate(
            n_ticks_to_generate=num_ticks_for_evaluation,
            target_datetimes=target_datetimes_for_eval,
        )

        # 5. Prepare initial window for the generator (if needed by your logic)
        #    This might be zeros or based on some real data segment.
        decoder_input_window_size = self.generator.params.get("decoder_input_window_size")
        num_all_features_gen = len(self.generator.params.get("full_feature_names_ordered", []))

        if num_all_features_gen == 0:
            raise ValueError("OptimizerPlugin: Generator's 'full_feature_names_ordered' is empty or not set.")

        initial_window_for_generator = np.zeros((decoder_input_window_size, num_all_features_gen), dtype=np.float32)
        # Optionally, populate initial_window_for_generator from a piece of real data if appropriate for optimization eval

        # 6. Call GeneratorPlugin.generate
        generated_full_sequence_batch = self.generator.generate(
            feeder_outputs_sequence=feeder_outputs_sequence,
            sequence_length_T=num_ticks_for_evaluation,
            initial_full_feature_window=initial_window_for_generator,
        )

        synthetic_data_np = generated_full_sequence_batch[0]  # Shape: (sequence_length_T, num_all_features)

        # 7. Evaluate the generated synthetic_data
        #    You'll need a segment of real data for comparison.
        #    This real data segment should be loaded or accessible here.
        #    For example, from self.evaluator.params.get('real_data_file') or preloaded.
        #    Let's assume `real_data_segment_for_eval_np` and `eval_feature_names_for_opt` are available.

        # Placeholder for real_data_segment_for_eval_np and eval_feature_names_for_opt
        # These need to be properly sourced (e.g., from preprocessed data accessible to the optimizer)
        # real_data_segment_for_eval_np = ...
        # eval_feature_names_for_opt = ...

        # Example: (This is highly dependent on how your optimizer gets its evaluation data)
        # if self.evaluator and hasattr(self.evaluator, '_load_real_data_for_evaluation'):
        #     real_df_eval, eval_feature_names_for_opt = self.evaluator._load_real_data_for_evaluation(num_ticks_for_evaluation)
        #     real_data_segment_for_eval_np = real_df_eval[eval_feature_names_for_opt].values
        # else:
        #     raise RuntimeError("OptimizerPlugin cannot access real data for evaluation.")

        # Align synthetic data columns with real data columns for evaluation
        # aligned_synthetic_df, aligned_real_df, aligned_features = self._align_data_for_evaluation(
        #     pd.DataFrame(synthetic_data_np, columns=self.generator.params.get("full_feature_names_ordered")),
        #     pd.DataFrame(real_data_segment_for_eval_np, columns=eval_feature_names_for_opt), # Assuming this is available
        #     eval_feature_names_for_opt # Target feature set
        # )

        # metrics = self.evaluator.evaluate(
        #     synthetic_data=aligned_synthetic_df.values,
        #     real_data_processed=aligned_real_df.values,
        #     feature_names=aligned_features,
        #     config=self.config # Pass the main config
        # )

        # Extract a fitness score from the metrics
        # fitness_score = metrics.get("overall_multivariate_fidelity", {}).get("avg_mmd_rbf", float('inf'))
        # if fitness_score is None or not isinstance(fitness_score, (int, float)) or np.isnan(fitness_score):
        #    fitness_score = float('inf') # Higher is worse for MMD

        # return (fitness_score,) # DEAP expects a tuple of fitness values

        # For now, to just fix the TypeError, the critical part is the feeder call.
        # The rest of the evaluation logic needs to be correctly implemented.
        # Returning a dummy fitness until the full evaluation flow is sorted.
        print(f"OptimizerPlugin: eval_individual called. Feeder generated {len(feeder_outputs_sequence)} outputs.")
        # This function must return a tuple (fitness_value,)
        return (0.0,)  # Placeholder fitness

    # ... other methods of OptimizerPlugin ...

    # You might need a helper to align data if features differ or order matters for evaluation
    # def _align_data_for_evaluation(self, synthetic_df, real_df, target_feature_names):
    #     common_features = [f for f in target_feature_names if f in synthetic_df.columns and f in real_df.columns]
    #     if not common_features:
    #         raise ValueError("Optimizer evaluation: No common features between synthetic and real data based on target_feature_names.")
    #     return synthetic_df[common_features], real_df[common_features], common_features
