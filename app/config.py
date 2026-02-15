"""Default configuration for synthetic-datagen."""

DEFAULT_VALUES = {
    # --- Mode ---
    "mode": "train",  # train | generate | optimize | evaluate

    # --- Plugin selection ---
    "trainer": "vae_gan_trainer",
    "generator": "typical_price_generator",
    "evaluator": "predictive_evaluator",
    "optimizer": "ga_optimizer",

    # --- Data ---
    "train_data": [],          # list of CSV paths for training
    "real_data": None,         # real CSV for evaluation
    "synthetic_data": None,    # synthetic CSV for evaluation
    "output_file": "synthetic_typical_price.csv",
    "metrics_file": "metrics.json",
    "real_train": None,        # real training CSV (d4) for evaluate mode
    "real_val": None,          # real validation CSV (d5) for evaluate mode
    "real_test": None,         # real test CSV (d6) for evaluate mode
    "predictor_dir": None,     # path to external predictor repo (optional)
    "eval_epochs": 50,         # epochs for built-in evaluation predictor
    "eval_batch_size": 64,     # batch size for built-in evaluation predictor
    "horizon": 1,              # prediction horizon for evaluation

    # --- Model I/O ---
    "save_model": "model.keras",
    "load_model": None,        # path to pre-trained model for generate mode

    # --- Training hyper-parameters (reference: phase_4_2) ---
    "window_size": 144,        # 144 × 4 h = 24 days
    "batch_size": 128,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "latent_dim": 16,
    "activation": "tanh",
    "intermediate_layers": 2,
    "initial_layer_size": 48,
    "layer_size_divisor": 2,
    "kl_weight": 1e-3,
    "kl_anneal_epochs": 40,
    "mmd_lambda": 1e-2,
    "l2_reg": 1e-6,
    "use_returns": True,       # model log-returns, not raw prices
    "early_patience": 120,
    "start_from_epoch": 15,
    "min_delta": 1e-7,

    # --- GAN-specific ---
    "discriminator_lr": 1e-4,
    "generator_lr": 1e-4,
    "disc_layers": [64, 32],
    "disc_dropout": 0.3,

    # --- Generation ---
    "n_samples": 5000,
    "seed": 42,
    "start_datetime": "2020-01-01 00:00:00",
    "interval_hours": 4,       # output interval (4 h)

    # --- Optimizer (staged incremental GA) ---
    "population_size": 20,
    "n_generations": 10,
    "crossover_prob": 0.7,
    "mutation_prob": 0.2,
    "optimization_patience": 4,
    "optimization_incremental": True,
    "optimization_increment_size": 1,
    "optimization_resume": False,
    "optimization_resume_file": "optimization_resume.json",
    "optimization_parameters_file": "best_params.json",
    "optimization_meta_mode": True,
    "meta_training_log": None,
    "deterministic_training": True,
    "random_seed": 42,
    "train_epochs": None,           # epochs per candidate (default: epochs//4)
    "predictor_epochs": 100,        # epochs for augmentation predictor test
    "n_synthetic_samples": 2190,    # 1 year of 4h data
    "hyperparameter_bounds": {
        "latent_dim":           [4, 64],
        "intermediate_layers":  [1, 4],
        "initial_layer_size":   [16, 128],
        "window_size":          [48, 288],
        "discriminator_lr":     [1e-5, 1e-2],
        "generator_lr":         [1e-5, 1e-2],
        "disc_dropout":         [0.0, 0.5],
        "learning_rate":        [1e-5, 1e-2],
        "activation":           [0, 7],
        "batch_size":           [32, 256],
        "l2_reg":               [1e-7, 1e-3],
        "kl_weight":            [1e-5, 1e-1],
        "kl_anneal_epochs":     [5, 50],
        "mmd_lambda":           [1e-4, 1e-1],
        "early_patience":       [20, 200],
        "min_delta":            [1e-10, 1e-6],
    },
    "optimization_stages": [
        {
            "stage": 1,
            "name": "Architecture",
            "description": "Core VAE-GAN structure",
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
    ],

    # --- Config I/O ---
    "load_config": None,
    "save_config": None,
    "log_level": "INFO",
}
