"""
Genetic-algorithm hyper-parameter optimiser plugin.

Uses DEAP to search over: latent_dim, learning_rate, kl_weight, mmd_lambda,
initial_layer_size, intermediate_layers.

Fitness = quality_score from DistributionEvaluator (minimise).
"""

from __future__ import annotations

import copy
import logging
import random
import tempfile
from typing import Any, Dict, Optional

import numpy as np
from deap import algorithms, base, creator, tools

from app.plugin_loader import load_plugin

log = logging.getLogger(__name__)

# DEAP creator (module-level, idempotent guard)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

# Search space
SPACE = {
    "latent_dim":          (4, 64),
    "learning_rate":       (1e-5, 1e-2),
    "kl_weight":           (1e-5, 1e-1),
    "mmd_lambda":          (1e-4, 1e-1),
    "initial_layer_size":  (16, 128),
    "intermediate_layers": (1, 4),
}
KEYS = list(SPACE.keys())


def _random_individual():
    genes = []
    for k in KEYS:
        lo, hi = SPACE[k]
        if isinstance(lo, int) and isinstance(hi, int):
            genes.append(random.randint(lo, hi))
        else:
            genes.append(10 ** random.uniform(np.log10(lo), np.log10(hi)))
    return creator.Individual(genes)


def _mutate(ind, indpb=0.3):
    for i, k in enumerate(KEYS):
        if random.random() < indpb:
            lo, hi = SPACE[k]
            if isinstance(lo, int) and isinstance(hi, int):
                ind[i] = random.randint(lo, hi)
            else:
                ind[i] = 10 ** random.uniform(np.log10(lo), np.log10(hi))
    return (ind,)


class GaOptimizer:
    """Plugin: GA hyper-parameter search for the trainer."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {}
        if config:
            self.cfg.update(config)

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    def optimize(self) -> Dict[str, Any]:
        cfg = self.cfg
        pop_size = cfg.get("population_size", 20)
        n_gen = cfg.get("n_generations", 50)

        toolbox = base.Toolbox()
        toolbox.register("individual", _random_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", _mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        log.info(f"Starting GA: pop={pop_size}, generations={n_gen}")
        algorithms.eaSimple(
            pop, toolbox,
            cxpb=cfg.get("crossover_prob", 0.7),
            mutpb=cfg.get("mutation_prob", 0.2),
            ngen=n_gen,
            halloffame=hof,
            verbose=True,
        )

        best = hof[0]
        best_params = {k: best[i] for i, k in enumerate(KEYS)}
        best_params["latent_dim"] = int(best_params["latent_dim"])
        best_params["intermediate_layers"] = int(best_params["intermediate_layers"])
        best_params["initial_layer_size"] = int(best_params["initial_layer_size"])

        log.info(f"Best fitness={best.fitness.values[0]:.6f}")
        log.info(f"Best params: {best_params}")
        return best_params

    def _evaluate_individual(self, ind) -> tuple[float]:
        """Train with candidate params → evaluate → return fitness."""
        trial_cfg = copy.deepcopy(self.cfg)
        for i, k in enumerate(KEYS):
            trial_cfg[k] = int(ind[i]) if isinstance(SPACE[k][0], int) else ind[i]

        # Use a temp file for the model
        with tempfile.TemporaryDirectory() as td:
            trial_cfg["save_model"] = f"{td}/model.keras"
            trial_cfg["output_file"] = f"{td}/synthetic.csv"
            trial_cfg["epochs"] = max(trial_cfg["epochs"] // 4, 20)  # faster

            try:
                trainer_cls = load_plugin("sdg.trainer", trial_cfg["trainer"])
                trainer = trainer_cls(trial_cfg)
                trainer.train()

                # Quick generation for evaluation
                trial_cfg["load_model"] = trial_cfg["save_model"]
                trial_cfg["n_samples"] = 2000
                gen_cls = load_plugin("sdg.generator", trial_cfg["generator"])
                gen = gen_cls(trial_cfg)
                gen.generate()

                # Evaluate against first training file as reference
                trial_cfg["synthetic_data"] = trial_cfg["output_file"]
                trial_cfg["real_data"] = trial_cfg["train_data"][0]
                eval_cls = load_plugin("sdg.evaluator", trial_cfg["evaluator"])
                ev = eval_cls(trial_cfg)
                metrics = ev.evaluate()
                score = metrics.get("quality_score", 999.0)
            except Exception as e:
                log.warning(f"Trial failed: {e}")
                score = 999.0

        return (score,)
