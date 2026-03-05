# BADA - RISC-V Throughput Prediction Framework

BADA is a framework for predicting RISC-V instruction throughput using machine learning models. It supports both real hardware experiments and simulator-based experiments with an iterative training approach.

## Overview

This project implements a RISC-V throughput prediction system with the following key features:

- **Multiple Model Types**: Support for Transformer, GNN, and LSTM models
- **Iterative Training**: Automated training loop with incremental learning
- **Dual Execution Modes**: 
  - Real hardware experiments via SSH connection
  - Simulator-based experiments using LLVM-MCA simulator
- **Automated Data Generation**: BO-based test case generation

## Project Structure

```
BADA/
├── main.py                 # Main entry point for training commands
├── hardware.sh             # Script for real hardware experiments
├── simulator.sh            # Script for simulator experiments
├── config/                 # Configuration files
├── models/                 # Model implementations
├── trainers/               # Training logic
├── scripts/                # Utility scripts (training, inference, preprocessing)
├── data/                   # Data processing utilities
├── bada/                   # BO and test generation tools
└── datasets/               # Training and validation datasets
```

## Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Execution Modes

### 1. Hardware Mode (`hardware.sh`)

The `hardware.sh` script is designed for experiments on real RISC-V hardware. It performs the following workflow:

**Features:**
- SSH connection to remote hardware board
- Automated file transfer (send test cases, retrieve results)
- Real-time execution on physical RISC-V processor
- Iterative training with hardware feedback

**Configuration:**
```bash
BOARD_IP="192.168.0.124"           # Board IP address
BOARD_USER="root"                   # Board username
BOARD_PASSWORD="starfive"           # Board password
BOARD_WORK_DIR="~/workdir"          # Working directory on board
```

**Usage:**
```bash
./hardware.sh --model_type transformer --train_data datasets/u74.json --epoch 30 --batch_size 8
```

**Options:**
- `--model_type`: Model architecture (transformer, gnn, lstm)
- `--train_data`: Path to training data JSON file
- `--val_data`: Path to validation data JSON file
- `--test_data`: Path to test data JSON file
- `--experiment_name`: Initial experiment name
- `--epoch`: Number of epochs
- `--batch_size`: Batch size
- `--incremental_exp_name`: Incremental experiment name
- `--no_improvement_limit`: Consecutive iterations without improvement before stopping

**Workflow:**
1. Run initial training
2. Generate test cases using BO
3. Send test cases to hardware board via SSH
4. Execute tests on real hardware
5. Retrieve execution results
6. Preprocess results into training data
7. Perform incremental training
8. Repeat steps 2-7 until convergence

### 2. Simulator Mode (`simulator.sh`)

The `simulator.sh` script is designed for experiments using the LLVM-MCA simulator. It provides a hardware-free alternative for development and testing.

**Features:**
- LLVM-MCA simulator integration
- No physical hardware required
- Faster iteration cycles
- Suitable for development and testing

**Usage:**
```bash
./simulator.sh --model_type transformer --train_data datasets/u74.json --epoch 30 --batch_size 8 --simulator sifive-u74
```

**Options:**
All options from hardware mode, plus:
- `--simulator`: LLVM-MCA simulator type (default: sifive-u74)

**Workflow:**
1. Run initial training
2. Generate test cases using fuzzing
3. Execute tests on LLVM-MCA simulator
4. Collect simulation results
5. Preprocess results into training data
6. Perform incremental training
7. Repeat steps 2-7 until convergence

## Iterative Training Process

Both execution modes implement an automated iterative training loop:

```
┌─────────────────────────────────────┐
│   Initial Training                  │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │  Generate     │◄──────────┐
       │  Test Cases   │           │
       └───────┬───────┘           │
               │                   │
               ▼                   │
    ┌──────────────────┐           │
    │ Execute on       │           │
    │ Hardware/Sim     │           │
    └────────┬─────────┘           │
             │                     │
             ▼                     │
    ┌──────────────────┐           │
    │ Preprocess       │           │
    │ Results          │           │
    └────────┬─────────┘           │
             │                     │
             ▼                     │
    ┌──────────────────┐           │
    │ Incremental      │           │
    │ Training         │           │
    └────────┬─────────┘           │
             │                     │
             ▼                     │
    ┌──────────────────┐           │
    │ Check            │           │
    │ Improvement?     │─── No ────┘
    └────────┬─────────┘
             │ Yes
             ▼
    ┌──────────────────┐
    │ Continue or      │
    │ Stop             │
    └──────────────────┘
```

The loop continues until:
- Validation loss stops improving for N consecutive iterations (configurable via `--no_improvement_limit`)
- The process is manually stopped

## Model Types

### Transformer
Default model type, suitable for sequence-to-sequence prediction tasks.

### GNN (Graph Neural Network)
Captures instruction dependencies as a graph structure.

### LSTM
Recurrent neural network for sequential data processing.

## Output and Logging

Experiments are saved in the `experiments` directory with the following structure:

```
experiments/
└── {experiment_name}_{timestamp}/
    ├── checkpoints/
    │   ├── model_best.pth
    │   └── checkpoint_epoch_*.pth
    ├── logs/
    │   └── experiment.log
    └── test_result.json
```

## Results Summary

Both scripts provide a detailed summary table at completion:

```
┌─────────────┬────────────────────────────────────┬─────────────┬─────────┬─────────────┐
│   Round     │         Experiment Name            │ Validation  │  Epoch  │   Test      │
│             │                                    │    Loss     │         │    Loss     │
├─────────────┼────────────────────────────────────┼─────────────┼─────────┼─────────────┤
│ Initial     │ transformer_20250305_123456        │ 0.123456    │ 15      │ 0.234567    │
│ Round 1     │ incre_20250305_124512              │ 0.112345    │ 12      │ 0.223456    │
│ Round 2     │ incre_20250305_125623              │ 0.102345 ★  │ 18      │ 0.213456    │
└─────────────┴────────────────────────────────────┴─────────────┴─────────┴─────────────┘
```

## Advanced Usage

### Custom Hardware Configuration

Edit the configuration section in `hardware.sh`:

```bash
BOARD_IP="your.board.ip"
BOARD_USER="your_username"
BOARD_PASSWORD="your_password"
BOARD_WORK_DIR="~/your_workdir"
```

### Custom Simulator

Specify different LLVM-MCA simulator models:

```bash
./simulator.sh --simulator your-custom-model
```

## Troubleshooting

### Hardware Connection Issues
- Verify board IP address and network connectivity
- Check SSH credentials and permissions
- Ensure working directory exists on the board

### Training Issues
- Check data file paths and formats
- Verify GPU/CPU availability
- Review experiment logs in `experiments/{experiment_name}/logs/`

### Simulator Issues
- Ensure LLVM-MCA is installed and accessible
- Verify simulator model compatibility
- Check `run_llvm_mca.sh` script permissions

## License

This project is provided as-is for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
