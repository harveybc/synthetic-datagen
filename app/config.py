"""Default configuration for synthetic-datagen."""

DEFAULT_VALUES = {
    # --- Mode ---
    "mode": "train",  # train | generate | optimize | evaluate

    # --- Plugin selection ---
    "trainer": "vae_gan_trainer",
    "generator": "typical_price_generator",
    "evaluator": "distribution_evaluator",
    "optimizer": "ga_optimizer",

    # --- Data ---
    "train_data": [],          # list of CSV paths for training
    "real_data": None,         # real CSV for evaluation
    "synthetic_data": None,    # synthetic CSV for evaluation
    "output_file": "synthetic_typical_price.csv",
    "metrics_file": "metrics.json",

    # --- Model I/O ---
    "save_model": "model.keras",
    "load_model": None,        # path to pre-trained model for generate mode

    # --- Training hyper-parameters (reference: phase_4_2) ---
    "window_size": 144,        # 144 × 4 h = 24 days
    "batch_size": 128,
    "epochs": 400,
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
    "downsample_factor": 1,    # 1 = already 4 h data; set 4 if training on 1 h

    # --- Optimizer (GA) ---
    "population_size": 20,
    "n_generations": 50,
    "crossover_prob": 0.7,
    "mutation_prob": 0.2,

    # --- Config I/O ---
    "load_config": None,
    "save_config": None,
    "log_level": "INFO",
}
