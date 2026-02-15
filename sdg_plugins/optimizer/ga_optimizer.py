"""
Staged incremental GA hyper-parameter optimizer for the synthetic data generator.

Mirrors predictor's optimization approach:
- DEAP-based genetic algorithm
- Staged optimization (optimize groups of related params sequentially)
- Each stage freezes previous best and only varies current stage params
- Fitness = augmentation test (does appending synthetic data improve predictor?)
- Resume support, meta-training logging, incremental stages

Fitness (MINIMIZE):
  augmented_val_mae - baseline_val_mae
  Negative = synthetic data helps → GOOD
  Positive = synthetic data hurts → BAD
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from deap import algorithms, base, creator, tools

from app.plugin_loader import load_plugin

log = logging.getLogger(__name__)

# DEAP creator (module-level, idempotent guard)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "SDGIndividual"):
    creator.create("SDGIndividual", list, fitness=creator.FitnessMin)


# ═══════════════════════════════════════════════════════════════════════════
# Default search space & stages
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_HYPERPARAMETER_BOUNDS = {
    # Stage 1: Architecture
    "latent_dim":           [4, 64],
    "intermediate_layers":  [1, 4],
    "initial_layer_size":   [16, 128],
    "window_size":          [48, 288],
    # Stage 2: GAN dynamics
    "discriminator_lr":     [1e-5, 1e-2],
    "generator_lr":         [1e-5, 1e-2],
    "disc_dropout":         [0.0, 0.5],
    # Stage 3: Training dynamics
    "learning_rate":        [1e-5, 1e-2],
    "activation":           [0, 7],    # mapped: 0=tanh,1=relu,2=elu,3=selu,4=swish,5=gelu,6=sigmoid,7=linear
    "batch_size":           [32, 256],
    # Stage 4: Regularization
    "l2_reg":               [1e-7, 1e-3],
    "kl_weight":            [1e-5, 1e-1],
    "kl_anneal_epochs":     [5, 50],
    "mmd_lambda":           [1e-4, 1e-1],
    # Stage 5: Convergence
    "early_patience":       [20, 200],
    "min_delta":            [1e-10, 1e-6],
}

DEFAULT_OPTIMIZATION_STAGES = [
    {
        "stage": 1,
        "name": "Architecture",
        "description": "Core VAE-GAN structure: latent dim, layer sizes, window",
        "parameters": ["latent_dim", "intermediate_layers", "initial_layer_size", "window_size"],
    },
    {
        "stage": 2,
        "name": "GAN Dynamics",
        "description": "Discriminator/generator learning rates and dropout",
        "parameters": ["discriminator_lr", "generator_lr", "disc_dropout"],
    },
    {
        "stage": 3,
        "name": "Training Dynamics",
        "description": "VAE learning rate, activation, batch size",
        "parameters": ["learning_rate", "activation", "batch_size"],
    },
    {
        "stage": 4,
        "name": "Regularization",
        "description": "KL weight, MMD, L2, annealing",
        "parameters": ["l2_reg", "kl_weight", "kl_anneal_epochs", "mmd_lambda"],
    },
    {
        "stage": 5,
        "name": "Convergence",
        "description": "Early stopping patience and delta",
        "parameters": ["early_patience", "min_delta"],
    },
]

ACTIVATION_MAP = {
    0: "tanh", 1: "relu", 2: "elu", 3: "selu",
    4: "swish", 5: "gelu", 6: "sigmoid", 7: "linear",
}

# Parameters that should be rounded to int
INT_PARAMS = {
    "latent_dim", "intermediate_layers", "initial_layer_size",
    "window_size", "batch_size", "kl_anneal_epochs", "early_patience",
}

# Parameters that should be sampled in log space
LOG_PARAMS = {
    "learning_rate", "discriminator_lr", "generator_lr",
    "l2_reg", "kl_weight", "mmd_lambda", "min_delta",
}


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer plugin
# ═══════════════════════════════════════════════════════════════════════════

class GaOptimizer:
    """Plugin: staged incremental GA hyper-parameter search.

    Config keys:
        hyperparameter_bounds: dict of {param: [lo, hi]}
        optimization_stages: list of stage dicts
        optimization_incremental: bool (default True)
        optimization_increment_size: int stages per run (default 1)
        optimization_resume: bool
        optimization_resume_file: path to resume JSON
        population_size: int (default 20)
        n_generations: int per stage (default 10)
        optimization_patience: int generations without improvement (default 4)
        crossover_prob: float (default 0.7)
        mutation_prob: float (default 0.2)
        deterministic_training: bool
        random_seed: int
        meta_training_log: path to CSV for meta-optimizer data
        train_epochs: epochs for VAE-GAN training per candidate (default epochs//4)
        predictor_epochs: epochs for predictor augmentation test (default 100)
        n_synthetic_samples: samples to generate per eval (default 2190 = 1 year 4h)

        # Augmentation evaluator config
        predictor_root: path
        d1_file, d2_file, d3_file: paths (d1=train gen, d2=val predictor, d3=test predictor)
        baseline_file: path to cached baseline results
    """

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {}
        if config:
            self.cfg.update(config)
        self._baseline: Optional[Dict[str, float]] = None

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    # ── public ──────────────────────────────────────────────────────────

    def optimize(self) -> Dict[str, Any]:
        """Run staged incremental optimization. Returns best params."""
        cfg = self.cfg

        bounds = cfg.get("hyperparameter_bounds", DEFAULT_HYPERPARAMETER_BOUNDS)
        stages = cfg.get("optimization_stages", DEFAULT_OPTIMIZATION_STAGES)
        incremental = cfg.get("optimization_incremental", True)
        increment_size = cfg.get("optimization_increment_size", 1)
        pop_size = cfg.get("population_size", 20)
        n_gen = cfg.get("n_generations", 10)
        patience = cfg.get("optimization_patience", 4)
        resume = cfg.get("optimization_resume", False)
        resume_file = cfg.get("optimization_resume_file", "optimization_resume.json")
        meta_log = cfg.get("meta_training_log")

        # Deterministic
        seed = cfg.get("random_seed", 42)
        if cfg.get("deterministic_training", True):
            random.seed(seed)
            np.random.seed(seed)

        # Load or compute baseline (once)
        self._ensure_baseline()

        # Resume state
        best_params = self._get_defaults()
        start_stage = 0
        if resume and os.path.exists(resume_file):
            state = self._load_resume(resume_file)
            best_params.update(state.get("best_params", {}))
            start_stage = state.get("completed_stages", 0)
            log.info(f"Resumed from stage {start_stage}, best fitness={state.get('best_fitness', '?')}")

        # Meta-training log
        meta_f = None
        if meta_log:
            os.makedirs(os.path.dirname(meta_log) or ".", exist_ok=True)
            write_header = not os.path.exists(meta_log)
            meta_f = open(meta_log, "a")
            if write_header:
                all_param_names = sorted(bounds.keys())
                meta_f.write("stage,generation,individual," + ",".join(all_param_names) + ",fitness\n")

        # Run stages
        if incremental:
            end_stage = min(start_stage + increment_size, len(stages))
            active_stages = stages[start_stage:end_stage]
        else:
            active_stages = stages
            start_stage = 0

        overall_best_fitness = float("inf")

        for stage_def in active_stages:
            stage_num = stage_def["stage"]
            stage_name = stage_def["name"]
            stage_params = stage_def["parameters"]

            # Filter to params that exist in bounds
            stage_params = [p for p in stage_params if p in bounds]
            if not stage_params:
                log.info(f"Stage {stage_num} ({stage_name}): no tunable params, skipping")
                continue

            log.info(f"\n{'='*60}")
            log.info(f"Stage {stage_num}: {stage_name}")
            log.info(f"Parameters: {stage_params}")
            log.info(f"Population: {pop_size}, Generations: {n_gen}")
            log.info(f"{'='*60}")

            stage_best, stage_fitness = self._run_stage(
                stage_params=stage_params,
                bounds=bounds,
                frozen_params=best_params,
                pop_size=pop_size,
                n_gen=n_gen,
                patience=patience,
                stage_num=stage_num,
                meta_f=meta_f,
            )

            # Update best params with stage results
            best_params.update(stage_best)

            if stage_fitness < overall_best_fitness:
                overall_best_fitness = stage_fitness

            log.info(f"Stage {stage_num} best: fitness={stage_fitness:.6f}")
            log.info(f"Stage {stage_num} params: {stage_best}")

            # Save resume state
            self._save_resume(resume_file, {
                "best_params": best_params,
                "best_fitness": overall_best_fitness,
                "completed_stages": stage_num,
            })

        if meta_f:
            meta_f.close()

        # Map activation index to string
        if "activation" in best_params:
            act_val = best_params["activation"]
            if isinstance(act_val, (int, float)):
                best_params["activation"] = ACTIVATION_MAP.get(int(round(act_val)), "tanh")

        # Round int params
        for k in INT_PARAMS:
            if k in best_params:
                best_params[k] = int(round(best_params[k]))

        log.info(f"\n{'='*60}")
        log.info(f"OPTIMIZATION COMPLETE")
        log.info(f"Best fitness (val_mae delta): {overall_best_fitness:.6f}")
        log.info(f"Best params: {json.dumps(best_params, indent=2, default=str)}")
        log.info(f"{'='*60}")

        # Save best params
        params_file = cfg.get("optimization_parameters_file", "best_params.json")
        os.makedirs(os.path.dirname(params_file) or ".", exist_ok=True)
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2, default=str)

        return best_params

    # ── stage runner ────────────────────────────────────────────────────

    def _run_stage(
        self,
        stage_params: List[str],
        bounds: Dict,
        frozen_params: Dict,
        pop_size: int,
        n_gen: int,
        patience: int,
        stage_num: int,
        meta_f=None,
    ) -> tuple[Dict[str, Any], float]:
        """Run GA for one stage. Returns (best_stage_params, best_fitness)."""

        keys = stage_params

        def make_individual():
            genes = []
            for k in keys:
                lo, hi = bounds[k]
                if k in INT_PARAMS:
                    genes.append(float(random.randint(int(lo), int(hi))))
                elif k in LOG_PARAMS:
                    genes.append(10 ** random.uniform(np.log10(lo), np.log10(hi)))
                else:
                    genes.append(random.uniform(lo, hi))
            return creator.SDGIndividual(genes)

        def mutate(ind, indpb=0.3):
            for i, k in enumerate(keys):
                if random.random() < indpb:
                    lo, hi = bounds[k]
                    if k in INT_PARAMS:
                        ind[i] = float(random.randint(int(lo), int(hi)))
                    elif k in LOG_PARAMS:
                        ind[i] = 10 ** random.uniform(np.log10(lo), np.log10(hi))
                    else:
                        ind[i] = random.uniform(lo, hi)
            return (ind,)

        def evaluate(ind):
            # Build trial params: frozen + this stage's genes
            trial = copy.deepcopy(frozen_params)
            for i, k in enumerate(keys):
                val = ind[i]
                if k in INT_PARAMS:
                    val = int(round(val))
                if k == "activation":
                    val = ACTIVATION_MAP.get(int(round(val)), "tanh")
                trial[k] = val

            fitness = self._evaluate_candidate(trial)
            return (fitness,)

        toolbox = base.Toolbox()
        toolbox.register("individual", make_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # Manual generation loop with patience
        best_fitness = float("inf")
        no_improve = 0

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        for gen in range(1, n_gen + 1):
            # Select + breed
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cfg.get("crossover_prob", 0.7):
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutation
            for mut in offspring:
                if random.random() < self.cfg.get("mutation_prob", 0.2):
                    toolbox.mutate(mut)
                    del mut.fitness.values

            # Evaluate invalids
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalids))
            for ind, fit in zip(invalids, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            record = stats.compile(pop)
            gen_best = record["min"]
            gen_avg = record["avg"]
            log.info(f"  Gen {gen:3d}/{n_gen} │ min={gen_best:.6f} avg={gen_avg:.6f}")

            # Meta-training log
            if meta_f:
                for idx, ind in enumerate(pop):
                    trial = copy.deepcopy(frozen_params)
                    for i, k in enumerate(keys):
                        trial[k] = ind[i]
                    all_keys = sorted(bounds.keys())
                    vals = [str(trial.get(k, "")) for k in all_keys]
                    meta_f.write(f"{stage_num},{gen},{idx}," + ",".join(vals) + f",{ind.fitness.values[0]}\n")
                meta_f.flush()

            # Patience check
            if gen_best < best_fitness - 1e-7:
                best_fitness = gen_best
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                log.info(f"  Early stopping at gen {gen} (no improvement for {patience} gens)")
                break

        # Extract best params for this stage
        best_ind = hof[0]
        best_stage_params = {}
        for i, k in enumerate(keys):
            val = best_ind[i]
            if k in INT_PARAMS:
                val = int(round(val))
            if k == "activation":
                val = ACTIVATION_MAP.get(int(round(val)), "tanh")
            best_stage_params[k] = val

        return best_stage_params, best_ind.fitness.values[0]

    # ── fitness evaluation ──────────────────────────────────────────────

    def _evaluate_candidate(self, params: Dict[str, Any]) -> float:
        """Train VAE-GAN with candidate params → generate → augmentation test.

        Returns fitness = augmented_val_mae - baseline_val_mae
        Negative = synthetic helps (GOOD), positive = hurts (BAD).
        """
        cfg = self.cfg
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.update(params)

        # Reduce training for speed during optimization
        train_epochs = cfg.get("train_epochs", max(cfg.get("epochs", 400) // 4, 50))
        trial_cfg["epochs"] = train_epochs
        n_samples = cfg.get("n_synthetic_samples", 2190)

        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.keras")
            synthetic_path = os.path.join(td, "synthetic.csv")
            trial_cfg["save_model"] = model_path
            trial_cfg["output_file"] = synthetic_path

            try:
                # 1. Train VAE-GAN
                trainer_cls = load_plugin("sdg.trainer", trial_cfg.get("trainer", "vae_gan_trainer"))
                trainer = trainer_cls(trial_cfg)
                trainer.train()

                # 2. Generate synthetic data
                trial_cfg["load_model"] = model_path
                trial_cfg["n_samples"] = n_samples
                gen_cls = load_plugin("sdg.generator", trial_cfg.get("generator", "typical_price_generator"))
                gen = gen_cls(trial_cfg)
                gen.generate(output_file=synthetic_path)

                # 3. Augmentation test
                augmented = self._run_augmentation_test(synthetic_path)
                fitness = augmented["val_mae"] - self._baseline["val_mae"]

                log.info(f"  Candidate: val_mae={augmented['val_mae']:.6f} "
                        f"(baseline={self._baseline['val_mae']:.6f}, "
                        f"delta={fitness:.6f})")

            except Exception as e:
                log.warning(f"  Candidate failed: {e}")
                fitness = 999.0

        return fitness

    def _ensure_baseline(self):
        """Load or compute the baseline (real data only) predictor performance."""
        baseline_file = self.cfg.get("baseline_file", "baseline_results.json")

        if self._baseline is not None:
            return

        if os.path.exists(baseline_file):
            log.info(f"Loading cached baseline from {baseline_file}")
            with open(baseline_file) as f:
                self._baseline = json.load(f)
            return

        log.info("Computing baseline (real data only) — runs ONCE")
        self._baseline = self._run_augmentation_test(synthetic_path=None)

        os.makedirs(os.path.dirname(baseline_file) or ".", exist_ok=True)
        with open(baseline_file, "w") as f:
            json.dump(self._baseline, f, indent=2)
        log.info(f"Baseline saved: val_mae={self._baseline['val_mae']:.6f}, "
                f"test_mae={self._baseline['test_mae']:.6f}")

    def _run_augmentation_test(self, synthetic_path: Optional[str]) -> Dict[str, float]:
        """Run predictor training with optional synthetic augmentation.

        Returns {"val_mae": float, "test_mae": float}
        """
        cfg = self.cfg

        # Try using the augmentation evaluator plugin
        try:
            eval_cfg = copy.deepcopy(cfg)
            if synthetic_path:
                eval_cfg["synthetic_data"] = synthetic_path

            eval_cls = load_plugin("sdg.evaluator", "augmentation_evaluator")
            evaluator = eval_cls(eval_cfg)

            if synthetic_path:
                result = evaluator._train_predictor(
                    augment_data=__import__("pandas").read_csv(synthetic_path, parse_dates=["DATE_TIME"])
                )
            else:
                result = evaluator._train_predictor(augment_data=None)

            return result

        except Exception as e:
            log.warning(f"Augmentation evaluator failed: {e}")
            # Fallback: use distribution evaluator score
            return {"val_mae": 999.0, "test_mae": 999.0}

    # ── defaults ────────────────────────────────────────────────────────

    def _get_defaults(self) -> Dict[str, Any]:
        """Get current config values as starting point."""
        defaults = {}
        bounds = self.cfg.get("hyperparameter_bounds", DEFAULT_HYPERPARAMETER_BOUNDS)
        for k in bounds:
            if k in self.cfg:
                defaults[k] = self.cfg[k]
        return defaults

    # ── resume ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_resume(path: str) -> Dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _save_resume(path: str, state: Dict):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
