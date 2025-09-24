#!/usr/bin/env python3
"""
Convenience script to run RFdiffusion pipeline with config file support
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from rfdiffusion_local import RFdiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        sys.exit(1)

def run_example(example_name: str, config: dict, args: argparse.Namespace):
    """Run a predefined example from config"""
    if example_name not in config.get('examples', {}):
        logger.error(f"Example '{example_name}' not found in config")
        logger.info(f"Available examples: {list(config.get('examples', {}).keys())}")
        sys.exit(1)

    example = config['examples'][example_name]
    defaults = config.get('defaults', {})

    logger.info(f"Running example: {example_name}")
    logger.info(f"Description: {example.get('description', 'No description')}")

    # Override defaults with example settings
    for key, value in example.items():
        if key != 'description':
            setattr(args, key.replace('_', '-'), value)

    # Set defaults for any missing values
    for key, value in defaults.items():
        attr_name = key.replace('_', '-')
        if not hasattr(args, attr_name) or getattr(args, attr_name) is None:
            setattr(args, attr_name, value)

def main():
    parser = argparse.ArgumentParser(description="Run RFdiffusion pipeline with config support")

    # Config and example options
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--example",
                        help="Run predefined example from config")
    parser.add_argument("--list-examples", action="store_true",
                        help="List available examples")

    # Container paths (can be overridden)
    parser.add_argument("--rfdiffusion-container",
                        help="Path to RFdiffusion singularity container")
    parser.add_argument("--colabfold-container",
                        help="Path to ColabFold singularity container")

    # RFdiffusion arguments
    parser.add_argument("--contigs",
                        help="Contig specification")
    parser.add_argument("--name",
                        help="Output name")
    parser.add_argument("--pdb",
                        help="Input PDB file or code")
    parser.add_argument("--iterations", type=int,
                        help="Number of diffusion iterations")
    parser.add_argument("--num-designs", type=int,
                        help="Number of designs to generate")
    parser.add_argument("--symmetry", choices=["none", "cyclic", "dihedral"],
                        help="Symmetry type")
    parser.add_argument("--order", type=int,
                        help="Symmetry order")
    parser.add_argument("--hotspot",
                        help="Hotspot residues")
    parser.add_argument("--chains",
                        help="Specific chains to use")

    # Validation arguments
    parser.add_argument("--num-seqs", type=int,
                        help="Number of sequences to generate")
    parser.add_argument("--num-recycles", type=int,
                        help="Number of AF2 recycles")
    parser.add_argument("--initial-guess", action="store_true",
                        help="Use initial guess for AF2")
    parser.add_argument("--use-multimer", action="store_true",
                        help="Use AF2 multimer")
    parser.add_argument("--rm-aa",
                        help="Amino acids to remove")
    parser.add_argument("--mpnn-sampling-temp", type=float,
                        help="MPNN sampling temperature")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip AF2 validation")

    # Other options
    parser.add_argument("--work-dir",
                        help="Working directory")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # List examples if requested
    if args.list_examples:
        logger.info("Available examples:")
        for name, example in config.get('examples', {}).items():
            logger.info(f"  {name}: {example.get('description', 'No description')}")
        sys.exit(0)

    # Run example if specified
    if args.example:
        run_example(args.example, config, args)

    # Set container paths from config if not provided
    if not args.rfdiffusion_container:
        args.rfdiffusion_container = config.get('containers', {}).get('rfdiffusion')
    if not args.colabfold_container:
        args.colabfold_container = config.get('containers', {}).get('colabfold')

    # Apply defaults for any unset values
    defaults = config.get('defaults', {})
    for key, value in defaults.items():
        attr_name = key.replace('_', '-')
        if not hasattr(args, attr_name) or getattr(args, attr_name) is None:
            setattr(args, attr_name, value)

    # Check required arguments
    if not args.rfdiffusion_container:
        logger.error("RFdiffusion container path not specified")
        logger.info("Set it in config.yaml or use --rfdiffusion-container")
        sys.exit(1)
    if not args.colabfold_container:
        logger.error("ColabFold container path not specified")
        logger.info("Set it in config.yaml or use --colabfold-container")
        sys.exit(1)
    if not args.contigs:
        logger.error("Contigs specification required")
        logger.info("Use --contigs or specify an example with --example")
        sys.exit(1)

    # Convert back to underscores for the pipeline
    pipeline_args = {}
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'example', 'list_examples']:
            pipeline_args[key.replace('-', '_')] = value

    try:
        # Initialize pipeline
        pipeline = RFdiffusionPipeline(
            singularity_rfdiffusion_path=pipeline_args['rfdiffusion_container'],
            singularity_colabfold_path=pipeline_args['colabfold_container'],
            work_dir=pipeline_args.get('work_dir', './outputs')
        )

        # Run RFdiffusion
        logger.info("Starting RFdiffusion...")
        output_path, contigs = pipeline.run_rfdiffusion(
            contigs=pipeline_args['contigs'],
            name=pipeline_args.get('name', 'test'),
            pdb=pipeline_args.get('pdb', ''),
            iterations=pipeline_args.get('iterations', 50),
            num_designs=pipeline_args.get('num_designs', 1),
            symmetry=pipeline_args.get('symmetry', 'none'),
            order=pipeline_args.get('order', 1),
            hotspot=pipeline_args.get('hotspot', ''),
            chains=pipeline_args.get('chains', '')
        )

        logger.info(f"RFdiffusion completed. Output: {output_path}")

        # Run validation unless skipped
        if not pipeline_args.get('skip_validation', False):
            logger.info("Starting validation...")
            best_structure = pipeline.run_proteinmpnn_and_alphafold(
                output_path=output_path,
                contigs=contigs,
                num_seqs=pipeline_args.get('num_seqs', 8),
                num_recycles=pipeline_args.get('num_recycles', 1),
                initial_guess=pipeline_args.get('initial_guess', False),
                use_multimer=pipeline_args.get('use_multimer', False),
                rm_aa=pipeline_args.get('rm_aa', 'C'),
                mpnn_sampling_temp=pipeline_args.get('mpnn_sampling_temp', 0.1),
                num_designs=pipeline_args.get('num_designs', 1)
            )
            logger.info(f"Best validated structure: {best_structure}")
        else:
            logger.info(f"Generated structure: {output_path}_0.pdb")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()