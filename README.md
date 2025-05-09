# Synthetic Data Generator (SDG)

## Description

The Synthetic Data Generator (SDG) project provides a plugin-based architecture to generate, evaluate, and optimize synthetic time-series data using configurable latent-space models. Key features include:

- **Plugin-based design**: Extend or replace components via plugins for latent sampling, generation, evaluation, and hyperparameter optimization.
- **Reproducibility**: Full control over configuration files, remote config support, and logging.
- **Separation of concerns**: Distinct plugins for each pipeline stage, orchestrated by a central DataProcessor.

## Installation Instructions

To install and set up the SDG application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/synthetic-datagen.git
    cd sdg
    ```

2. **Create and Activate a Virtual Environment** (Python 3.8+ is required):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate   # On Windows
    ```

3. **Install the Package**:
    ```bash
    pip install --upgrade pip
    pip install .
    ```

4. **(Optional) Install Development Extras**:
    ```bash
    pip install "sdg[dev]"
    ```

## Usage

SDG exposes a command-line interface `sdg` with the following usage pattern:

```bash
sdg --feeder default_feeder --generator default_generator --n_samples 1000 \
    --latent_dim 16 --batch_size 32 --decoder_model_file examples/results/phase_4_1/decoder_model.keras \
    --real_data_file examples/data/phase_4_1/normalized_d1.csv \
    --output_file examples/results/phase_4_1/synthetic_data.csv \
    --metrics_file examples/results/phase_4_1/evaluation_metrics.json \
    --latent_dim_range 8 64 --iterations 10 \
    --load_config examples/config/phase_4_1/sdg_config.json \
    --save_config examples/results/phase_4_1/config_out.json \
    --save_log examples/results/phase_4_1/debug_out.json
```

For a full list of command-line options and their descriptions, see the **Parameters** section below or run:

```bash
sdg --help
```

## Parameters

| Parameter                 | Type    | Description                                                                                  | Default                                                          |
|---------------------------|---------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `--feeder`                | string  | Name of the feeder plugin to use.                                                            | `default_feeder`                                                 |
| `--generator`             | string  | Name of the generator plugin to use.                                                         | `default_generator`                                              |
| `--evaluator`             | string  | Name of the evaluator plugin to use.                                                         | `default_evaluator`                                              |
| `--optimizer`             | string  | Name of the optimizer plugin to use.                                                         | `default_optimizer`                                              |
| `--n_samples`, `-ns`      | int     | Number of synthetic samples (windows) to generate.                                           | `1000`                                                           |
| `--latent_dim`, `-ld`     | int     | Dimension of the latent space to use for generation.                                         | `16`                                                             |
| `--batch_size`, `-bs`     | int     | Batch size for the generation process.                                                       | `32`                                                             |
| `--decoder_model_file`    | string  | Path to the pretrained decoder model file.                                                   | `examples/results/phase_4_1/decoder_model.keras`                 |
| `--real_data_file`, `-rdf`| string  | Path to the CSV file with real reference data for evaluation.                                | `examples/data/phase_4_1/normalized_d1.csv`                      |
| `--output_file`, `-of`    | string  | Path to the output CSV file for synthetic data.                                              | `examples/results/phase_4_1/synthetic_data.csv`                  |
| `--metrics_file`, `-mf`   | string  | Path to the output file for evaluation metrics (JSON or CSV).                                | `examples/results/phase_4_1/evaluation_metrics.json`             |
| `--latent_dim_range`, `-le` | int[]  | Min and max latent dimensions for optimization.                                              | `[8, 64]`                                                        |
| `--iterations`, `-it`     | int     | Number of optimizer iterations (generations).                                                | `10`                                                             |
| `--remote_log`, `-rl`     | string  | URL of remote API endpoint for saving debug information.                                     | `null`                                                           |
| `--remote_load_config`, `-rlc` | string | URL of remote JSON configuration file to download and execute.                            | `null`                                                           |
| `--remote_save_config`, `-rsc` | string | URL of remote API endpoint to save configuration in JSON format.                        | `null`                                                           |
| `--username`, `-u`        | string  | Username for the API endpoint.                                                               | `null`                                                           |
| `--password`, `-p`        | string  | Password for the API endpoint.                                                               | `null`                                                           |
| `--load_config`, `-lc`    | string  | Path to load a configuration file.                                                           | `examples/config/phase_4_1/sdg_config.json`                      |
| `--save_config`, `-sc`    | string  | Path to save the current configuration.                                                      | `examples/results/phase_4_1/config_out.json`                     |
| `--save_log`, `-sl`       | string  | Path to save the current debug information.                                                  | `examples/results/phase_4_1/debug_out.json`                      |
| `--quiet_mode`, `-qm`     | boolean | Suppress output messages.                                                                    | `False`                                                          |

## Directory Structure

```
sdg/
│
├── sdg_plugins/                       # Plugin implementations
│   ├── feeder_plugin.py               # FeederPlugin
│   ├── generator_plugin.py            # GeneratorPlugin
│   ├── evaluator_plugin.py            # EvaluatorPlugin
│   └── optimizer_plugin.py            # OptimizerPlugin
│
├── examples/                          # Example files
│   ├── config/                        # Example configuration files (phase_4_1)
│   ├── data/                          # Example data (phase_4_1)
│   ├── results/                       # Example results (phase_4_1)
│   └── scripts/                       # Example execution scripts
│
├── app/                    # Core orchestration package
│   └── main.py                        # Entry point for sdg CLI
│
├── setup.py                           # Package installation script
├── cli.py                             # Command-line interface definitions
├── config.py                          # Default configuration values
└── README.md                          # This file
```

## Example Plugin Model

```mermaid
graph TD

    subgraph SDG_Feeder ["Feeder: Generate Latent Codes"]
        A[Latent Dim & Samples] --> B{"FeederPlugin"};
        B --> C[(Z: latent codes)];
    end

    subgraph SDG_Generator ["Generator: Decode to Windows"]
        C --> D{"GeneratorPlugin"};
        D --> E[(X_syn: windows)];
    end

    subgraph SDG_Evaluator ["Evaluator: Compute Metrics"]
        E --> F{"EvaluatorPlugin"};
        F --> G[(Metrics & Reports)];
    end

    subgraph SDG_Optimizer ["Optimizer: Tune Hyperparams"]
        H[Search Space] --> I{"OptimizerPlugin"};
        I --> J[(Best Hyperparameters)];
    end

    E --> I  # downstream evaluation in optimization

    style B,C,D,E,F,G,I,J fill:#f9f,stroke:#333,stroke-width:1px;
```
