#!/usr/bin/env python3
"""
Generate Depo evaluation data from YAML config.

This script reads configuration from a YAML file and generates evaluation splits
for the Depo dataset. It replaces the shell script generate_eval_depo.sh to avoid
parameter duplication.

Usage:
    python scripts/generate_eval_depo.py --config lingua_modified/apps/main/configs/depo_debug.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Depo evaluation data from YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    # Allow command-line overrides for eval-specific parameters
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Override number of examples per hop distance"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Override number of nodes (default: from config eval.N or depo_generation_args.max_nodes)"
    )
    parser.add_argument(
        "--K-max",
        type=int,
        default=None,
        help="Override max hop distance (default: from config eval.K_max or depo_generation_args.max_hops)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract relevant sections
    depo_args = config.get('depo_generation_args', {})
    eval_config = config.get('eval', {})
    seed = config.get('seed', 42)
    
    # Get base output directory from eval config
    base_output_dir = eval_config.get('eval_dir')
    if not base_output_dir:
        print("Error: eval.eval_dir not found in config")
        sys.exit(1)
    
    # Get parameters with priority: CLI args > eval config > depo_generation_args > defaults
    N = args.N or eval_config.get('N') or depo_args.get('max_nodes', 125)
    K_max = args.K_max or eval_config.get('K_max') or depo_args.get('max_hops', 8)
    num_examples = args.num_examples or eval_config.get('num_examples_generate', 200)
    
    # These come from depo_generation_args (shared between training and eval)
    base_vocab_size = depo_args.get('base_vocab_size', 4)
    min_token_length = depo_args.get('min_token_length', 5)
    max_token_length = depo_args.get('max_token_length', 7)
    
    # Construct parameterized subfolder name
    subfolder = (
        f"N_{N}_K_{K_max}"
        f"_vs_{base_vocab_size}"
        f"_from_{min_token_length}_to_{max_token_length}"
    )
    output_dir = str(Path(base_output_dir) / subfolder)
    
    print("=" * 60)
    print("Generating Depo Evaluation Data")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Subfolder: {subfolder}")
    print(f"Output directory: {output_dir}")
    print(f"Examples per hop: {num_examples}")
    print(f"N (num nodes): {N}")
    print(f"K_max (max hops): {K_max}")
    print(f"Base vocab size: {base_vocab_size}")
    print(f"Token length: [{min_token_length}, {max_token_length}]")
    print(f"Seed: {seed}")
    print("=" * 60)
    print()
    
    # Clean up output directory if it exists
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Cleaning up existing directory: {output_dir}")
        shutil.rmtree(output_dir)
        print("✓ Directory cleaned")
        print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Import the generation function
    # Add the data_synthetic_pretrain directory to path for imports
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "data_synthetic_pretrain" / "Depo"))
    
    from generate_eval_jsonl import generate_eval_split
    
    # Generate evaluation splits for each hop distance
    hop_distance = 1
    while hop_distance <= K_max:
        eval_file_path = output_path / f"depo.eval.hop_{hop_distance}.jsonl"
        generate_eval_split(
            hop_distance=hop_distance,
            num_examples=num_examples,
            output_path=str(eval_file_path),
            N=N,
            K_max=K_max,
            base_vocab_size=base_vocab_size,
            min_token_length=min_token_length,
            max_token_length=max_token_length,
            seed=seed,
        )
        hop_distance *= 2
    
    print("=" * 60)
    print("✓ Evaluation data generation complete!")
    print("=" * 60)
    print("Generated files:")
    for f in sorted(output_path.glob("*.jsonl")):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    print()


if __name__ == "__main__":
    main()
