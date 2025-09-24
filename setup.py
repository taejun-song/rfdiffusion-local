#!/usr/bin/env python3
"""
Setup script to configure singularity container paths
"""

import os
import subprocess
import yaml
from pathlib import Path

def find_singularity_containers():
    """Try to find RFdiffusion and ColabFold singularity containers"""
    possible_paths = [
        "/usr/local/bin",
        "/opt/singularity",
        "/home/$USER/containers",
        "/data/containers",
        "~/containers",
        "."
    ]

    containers = {}

    print("Searching for singularity containers...")

    for path in possible_paths:
        expanded_path = os.path.expanduser(os.path.expandvars(path))
        if os.path.exists(expanded_path):
            print(f"Checking {expanded_path}...")

            # Look for RFdiffusion containers
            for pattern in ["*rfdiffusion*.sif", "*RFdiffusion*.sif", "*rf_diffusion*.sif"]:
                try:
                    result = subprocess.run(
                        ["find", expanded_path, "-name", pattern, "-type", "f"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        containers["rfdiffusion"] = result.stdout.strip().split('\n')[0]
                        print(f"Found RFdiffusion container: {containers['rfdiffusion']}")
                        break
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    continue

            # Look for ColabFold containers
            for pattern in ["*colabfold*.sif", "*ColabFold*.sif", "*colab_fold*.sif"]:
                try:
                    result = subprocess.run(
                        ["find", expanded_path, "-name", pattern, "-type", "f"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        containers["colabfold"] = result.stdout.strip().split('\n')[0]
                        print(f"Found ColabFold container: {containers['colabfold']}")
                        break
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    continue

    return containers

def update_config(containers):
    """Update config.yaml with found container paths"""
    config_path = "config.yaml"

    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update container paths
    if 'containers' not in config:
        config['containers'] = {}

    for container_type, path in containers.items():
        config['containers'][container_type] = path

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated {config_path} with container paths")

def manual_setup():
    """Manual setup if auto-detection fails"""
    print("\nAutomatic detection failed or incomplete.")
    print("Please provide the paths to your singularity containers:")

    containers = {}

    rfdiffusion_path = input("RFdiffusion container path: ").strip()
    if rfdiffusion_path and os.path.exists(rfdiffusion_path):
        containers["rfdiffusion"] = rfdiffusion_path
    else:
        print("Warning: RFdiffusion container path not found or invalid")

    colabfold_path = input("ColabFold container path: ").strip()
    if colabfold_path and os.path.exists(colabfold_path):
        containers["colabfold"] = colabfold_path
    else:
        print("Warning: ColabFold container path not found or invalid")

    return containers

def main():
    print("RFdiffusion Local Pipeline Setup")
    print("=" * 40)

    # Try automatic detection
    containers = find_singularity_containers()

    # If not all containers found, try manual setup
    if len(containers) < 2:
        print(f"\nFound {len(containers)}/2 containers automatically")
        if input("Try manual setup? (y/N): ").lower().startswith('y'):
            manual_containers = manual_setup()
            containers.update(manual_containers)

    if containers:
        update_config(containers)
        print(f"\nSetup complete! Found containers:")
        for container_type, path in containers.items():
            print(f"  {container_type}: {path}")
        print(f"\nYou can now run the pipeline with:")
        print(f"  python run_pipeline.py --example unconditional_monomer")
        print(f"  python run_pipeline.py --contigs 100 --name my_protein")
    else:
        print("\nNo containers found. Please:")
        print("1. Check that singularity containers are installed")
        print("2. Update config.yaml manually with correct paths")
        print("3. Or run this script again with correct paths")

if __name__ == "__main__":
    main()