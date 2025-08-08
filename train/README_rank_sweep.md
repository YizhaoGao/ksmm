# Rank Sweep Study for KS Linear Chain

This directory contains scripts for conducting rank sweep studies to analyze how the rank parameter affects training convergence in KS Linear Chain models.

## Scripts Overview

### 1. `train_with_random.py` (Modified)
The main training script that has been enhanced to save loss curves for analysis.

**Key additions:**
- Saves loss curves to JSON and CSV files for each training run
- Includes rank information in the saved data
- Compatible with the original command-line interface

### 2. `rank_sweep_study.py`
Orchestrates training across multiple rank values and generates comparison plots.

**Features:**
- Runs training for multiple ranks automatically
- Collects and aggregates results
- Generates loss curve comparison plots
- Saves summary statistics
- Handles failed training runs gracefully

### 3. `generate_test_weight.py`
Creates test weight matrices for experimentation.

**Features:**
- Configurable matrix dimensions
- Multiple initialization methods (Xavier, Kaiming, etc.)
- Supports different data types (float16, float32)

### 4. `plot_rank_results.py`
Re-plots results from saved rank sweep data.

**Features:**
- Load and visualize previously saved results
- Customizable plot parameters
- Summary statistics display

### 5. `example_rank_sweep.py`
Demonstrates the complete workflow with a simple example.

## Quick Start

### Option 1: Run the Example
```bash
cd /home/aiscuser/ksmm/train
python example_rank_sweep.py
```

This will:
1. Generate a test weight matrix (1024×2048)
2. Run rank sweep with ranks [2, 4, 8, 16, 32, 64]
3. Generate comparison plots
4. Save results to `./demo_rank_sweep_results/`

### Option 2: Manual Workflow

#### Step 1: Generate or Prepare Weight Matrix
```bash
# Generate a test weight matrix
python generate_test_weight.py \
    --output_features 2048 \
    --input_features 4096 \
    --output_path ./my_weight.pth \
    --dtype float32 \
    --init_method xavier_uniform
```

#### Step 2: Run Rank Sweep Study
```bash
# Run rank sweep with custom parameters
python rank_sweep_study.py \
    --weight_path ./my_weight.pth \
    --ranks 2 4 8 16 32 64 128 256 512 \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_samples 10000 \
    --output_dir ./rank_sweep_results \
    --save_prefix my_study \
    --log_scale
```

#### Step 3: Analyze Results
The rank sweep study automatically generates plots, but you can re-plot with:
```bash
python plot_rank_results.py \
    --results_dir ./rank_sweep_results \
    --save_prefix my_study \
    --log_scale
```

## Command Line Arguments

### `rank_sweep_study.py` Key Arguments

**Required:**
- `--weight_path`: Path to the dense weight matrix (.pth file)

**Rank Control:**
- `--ranks`: List of ranks to test (default: [2,4,8,16,32,64,128,256,512])
- `--max_rank`: Maximum rank to test (filters the ranks list)

**Training Parameters:**
- `--num_epochs`: Epochs per rank (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--num_samples`: Number of random samples (default: 10000)

**Output Control:**
- `--output_dir`: Results directory (default: ./rank_sweep_results)
- `--save_prefix`: File prefix (default: rank_sweep)
- `--log_scale`: Use log scale for loss plots

## Output Files

After running a rank sweep study, you'll find:

```
rank_sweep_results/
├── rank_2/
│   ├── rank_2_rank_2_loss_curves.csv
│   ├── rank_2_rank_2_loss_curves.json
│   └── rank_2_final.pth
├── rank_4/
│   └── ...
├── rank_sweep_loss_curves.png        # Main comparison plot
├── rank_sweep_final_losses.png       # Final loss comparison
├── rank_sweep_summary.csv            # Summary table
└── rank_sweep_summary.json           # Summary in JSON format
```

### Key Output Files

1. **Loss Curves Plot** (`*_loss_curves.png`): Side-by-side training and validation loss curves for all ranks
2. **Final Loss Plot** (`*_final_losses.png`): Bar chart comparing final losses across ranks
3. **Summary CSV** (`*_summary.csv`): Table with final and minimum losses for each rank
4. **Individual Loss Data** (`rank_X/*_loss_curves.csv`): Per-epoch loss data for each rank

## Example Results Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load summary results
df = pd.read_csv('./rank_sweep_results/rank_sweep_summary.csv')

# Display summary
print(df)

# Plot final validation loss vs rank
plt.figure(figsize=(10, 6))
plt.semilogx(df['rank'], df['final_val_loss'], 'o-')
plt.xlabel('Rank')
plt.ylabel('Final Validation Loss')
plt.title('Final Validation Loss vs Rank')
plt.grid(True)
plt.show()

# Find optimal rank
optimal_rank = df.loc[df['final_val_loss'].idxmin(), 'rank']
print(f"Optimal rank: {optimal_rank}")
```

## Tips and Best Practices

1. **Start Small**: Begin with fewer epochs and smaller ranks to test the workflow
2. **Memory Management**: Higher ranks require more GPU memory; reduce batch size if needed
3. **Time Estimation**: Each rank trains independently, so total time = num_ranks × time_per_rank
4. **Failed Runs**: The script continues even if some ranks fail; check logs for issues
5. **Reproducibility**: Use consistent random seeds across runs for fair comparison

## Troubleshooting

**Out of Memory Errors:**
- Reduce `--batch_size`
- Use `--dtype float16`
- Lower `--max_rank` to exclude very high ranks

**Training Failures:**
- Check that the weight matrix path is correct
- Verify CUDA availability for GPU training
- Ensure sufficient disk space for results

**Import Errors:**
- Ensure all dependencies are installed: `torch`, `matplotlib`, `pandas`, `numpy`, `tqdm`
- Check that the KS modules are properly installed and importable

## Customization

You can easily modify the scripts for your specific needs:

- **Different Rank Sequences**: Modify the `--ranks` argument
- **Custom Loss Functions**: Edit the criterion in `train_with_random.py`
- **Additional Metrics**: Add more metrics to the loss curve saving function
- **Plot Styling**: Customize colors, styles, and layouts in the plotting functions
