#!/usr/bin/env python3
"""
Test script to verify the pipeline setup without running containers
"""

import sys
from pathlib import Path
import yaml
from rfdiffusion_local import RFdiffusionPipeline

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import scipy
        import yaml
        from Bio import SeqIO
        print("✓ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test config file loading"""
    print("Testing config file...")
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        required_sections = ['containers', 'defaults', 'examples']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False

        print("✓ Config file loaded and validated")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_pipeline_init():
    """Test pipeline initialization with dummy paths"""
    print("Testing pipeline initialization...")
    try:
        # Use dummy paths for testing
        pipeline = RFdiffusionPipeline(
            singularity_rfdiffusion_path="/dummy/path/rfdiffusion.sif",
            singularity_colabfold_path="/dummy/path/colabfold.sif",
            work_dir="./test_outputs"
        )
        print("✗ Pipeline should have failed with dummy paths")
        return False
    except FileNotFoundError:
        print("✓ Pipeline correctly validates container paths")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_contig_fixing():
    """Test contig string parsing"""
    print("Testing contig parsing...")
    try:
        pipeline = RFdiffusionPipeline.__new__(RFdiffusionPipeline)

        # Test simple contigs
        result = pipeline.fix_contigs(["100"])
        assert result == ["100"], f"Expected ['100'], got {result}"

        # Test range contigs
        result = pipeline.fix_contigs(["50-100"])
        assert 50 <= int(result[0]) <= 100, f"Range result {result[0]} not in 50-100"

        print("✓ Contig parsing works correctly")
        return True
    except Exception as e:
        print(f"✗ Contig parsing error: {e}")
        return False

def main():
    print("RFdiffusion Local Pipeline - Setup Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_config,
        test_pipeline_init,
        test_contig_fixing
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("\n✓ Setup appears to be working correctly!")
        print("Next steps:")
        print("1. Update container paths in config.yaml")
        print("2. Run: python setup.py (to auto-detect containers)")
        print("3. Test with: uv run python run_pipeline.py --example unconditional_monomer --skip-validation")
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()