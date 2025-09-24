#!/usr/bin/env python3
"""
Local RFdiffusion Pipeline with AlphaFold2 validation
Adapted from ColabDesign notebook for local execution with singularity containers
"""

import os
import sys
import time
import random
import string
import subprocess
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RFdiffusionPipeline:
    def __init__(self,
                 singularity_rfdiffusion_path: str,
                 singularity_colabfold_path: str,
                 work_dir: str = "./outputs"):
        """
        Initialize RFdiffusion pipeline with singularity containers

        Args:
            singularity_rfdiffusion_path: Path to RFdiffusion singularity container
            singularity_colabfold_path: Path to ColabFold singularity container
            work_dir: Working directory for outputs
        """
        self.rfdiffusion_container = singularity_rfdiffusion_path
        self.colabfold_container = singularity_colabfold_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        # Validate container paths
        if not os.path.exists(self.rfdiffusion_container):
            raise FileNotFoundError(f"RFdiffusion container not found: {self.rfdiffusion_container}")
        if not os.path.exists(self.colabfold_container):
            raise FileNotFoundError(f"ColabFold container not found: {self.colabfold_container}")

    def fix_contigs(self, contigs: List[str], parsed_pdb=None) -> List[str]:
        """Fix contig format for RFdiffusion"""
        fixed_contigs = []
        for contig in contigs:
            if "-" in contig and contig.split("-")[0].isdigit():
                # Handle range like "50-100"
                start, end = map(int, contig.split("-"))
                length = random.randint(start, end)
                fixed_contigs.append(str(length))
            else:
                fixed_contigs.append(contig)
        return fixed_contigs

    def get_pdb_file(self, pdb_code: str, output_dir: Path) -> Optional[str]:
        """Download or locate PDB file"""
        if pdb_code == "" or pdb_code is None:
            return None

        if os.path.isfile(pdb_code):
            return pdb_code

        pdb_file = output_dir / f"{pdb_code}.pdb"

        if len(pdb_code) == 4:
            # Standard PDB code
            url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
            try:
                subprocess.run(["wget", "-q", "-O", str(pdb_file), url], check=True)
                logger.info(f"Downloaded PDB {pdb_code}")
                return str(pdb_file)
            except subprocess.CalledProcessError:
                logger.error(f"Failed to download PDB {pdb_code}")
                return None
        else:
            # AlphaFold model
            url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
            try:
                subprocess.run(["wget", "-q", "-O", str(pdb_file), url], check=True)
                logger.info(f"Downloaded AlphaFold model {pdb_code}")
                return str(pdb_file)
            except subprocess.CalledProcessError:
                logger.error(f"Failed to download AlphaFold model {pdb_code}")
                return None

    def run_rfdiffusion(self,
                       contigs: str,
                       name: str = "test",
                       pdb: str = "",
                       iterations: int = 50,
                       num_designs: int = 1,
                       symmetry: str = "none",
                       order: int = 1,
                       hotspot: str = "",
                       chains: str = "") -> Tuple[str, List[str]]:
        """
        Run RFdiffusion to generate protein backbone structures

        Args:
            contigs: Contig specification (e.g., "100", "A:50", "40/A163-181/40")
            name: Name for the output files
            pdb: PDB file path or code for conditional generation
            iterations: Number of diffusion iterations
            num_designs: Number of designs to generate
            symmetry: Symmetry type ("none", "cyclic", "dihedral")
            order: Symmetry order
            hotspot: Hotspot residues for binder design
            chains: Specific chains to use from input PDB

        Returns:
            Tuple of (output_path, fixed_contigs)
        """

        # Create unique output path
        path = name
        while (self.work_dir / f"{path}_0.pdb").exists():
            path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

        full_path = self.work_dir / path
        full_path.mkdir(exist_ok=True)

        logger.info(f"Running RFdiffusion with output path: {full_path}")

        # Parse contigs
        contigs_list = contigs.replace(",", " ").replace(":", " ").split()
        fixed_contigs = self.fix_contigs(contigs_list)

        # Determine mode
        is_fixed = any(contig.split("/")[0][0].isalpha() for contig in contigs_list for x in contig.split("/"))
        is_free = any(contig.split("-")[0].isdigit() for contig in contigs_list for x in contig.split("/"))

        if len(contigs_list) == 0 or not is_free:
            mode = "partial"
        elif is_fixed:
            mode = "fixed"
        else:
            mode = "free"

        logger.info(f"Mode: {mode}, Contigs: {fixed_contigs}")

        # Set up RFdiffusion command options
        opts = [
            f"inference.output_prefix={full_path}",
            f"inference.num_designs={num_designs}",
            f"'contigmap.contigs=[{' '.join(fixed_contigs)}]'",
            "inference.dump_pdb=True"
        ]

        # Handle input PDB if provided
        if pdb and mode in ["partial", "fixed"]:
            pdb_file = self.get_pdb_file(pdb, full_path)
            if pdb_file:
                opts.append(f"inference.input_pdb={pdb_file}")
                if mode == "partial":
                    iterations = int(80 * (iterations / 200))
                    opts.append(f"diffuser.partial_T={iterations}")
                else:
                    opts.append(f"diffuser.T={iterations}")
            else:
                logger.warning("Could not get PDB file, running in free mode")
                opts.append(f"diffuser.T={iterations}")
        else:
            opts.append(f"diffuser.T={iterations}")

        # Handle hotspot
        if hotspot:
            opts.append(f"ppi.hotspot_res=[{hotspot}]")

        # Handle symmetry
        if symmetry != "none":
            if symmetry == "cyclic":
                sym = f"c{order}"
                copies = order
            elif symmetry == "dihedral":
                sym = f"d{order}"
                copies = order * 2
            else:
                sym, copies = None, 1

            if sym:
                sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
                opts = sym_opts + opts
                fixed_contigs = fixed_contigs * copies

        # Build and run command
        opts_str = " ".join(opts)
        cmd = [
            "singularity", "exec",
            "--bind", f"{self.work_dir}:/outputs",
            self.rfdiffusion_container,
            "python", "/app/RFdiffusion/run_inference.py"
        ] + opts_str.split()

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.work_dir))
            if result.returncode != 0:
                logger.error(f"RFdiffusion failed: {result.stderr}")
                raise RuntimeError(f"RFdiffusion failed: {result.stderr}")
            else:
                logger.info("RFdiffusion completed successfully")
        except Exception as e:
            logger.error(f"Error running RFdiffusion: {e}")
            raise

        return str(full_path), fixed_contigs

    def run_proteinmpnn_and_alphafold(self,
                                     output_path: str,
                                     contigs: List[str],
                                     num_seqs: int = 8,
                                     num_recycles: int = 1,
                                     initial_guess: bool = False,
                                     use_multimer: bool = False,
                                     rm_aa: str = "C",
                                     mpnn_sampling_temp: float = 0.1,
                                     num_designs: int = 1) -> str:
        """
        Run ProteinMPNN for sequence design and AlphaFold2 for validation

        Args:
            output_path: Path to RFdiffusion output
            contigs: Fixed contigs from RFdiffusion
            num_seqs: Number of sequences to generate
            num_recycles: Number of AF2 recycles
            initial_guess: Use initial guess for AF2
            use_multimer: Use AF2 multimer
            rm_aa: Amino acids to remove
            mpnn_sampling_temp: MPNN sampling temperature
            num_designs: Number of designs

        Returns:
            Path to best validated structure
        """

        output_dir = Path(output_path)
        logger.info(f"Running ProteinMPNN and AlphaFold validation on {output_dir}")

        # Prepare ColabFold command
        contigs_str = ":".join(contigs)

        cmd = [
            "singularity", "exec",
            "--bind", f"{self.work_dir}:/workspace",
            self.colabfold_container,
            "python", "/app/colabdesign/rf/designability_test.py",
            f"--pdb={output_path}_0.pdb",
            f"--loc={output_path}",
            f"--contig={contigs_str}",
            f"--num_seqs={num_seqs}",
            f"--num_recycles={num_recycles}",
            f"--rm_aa={rm_aa}",
            f"--mpnn_sampling_temp={mpnn_sampling_temp}",
            f"--num_designs={num_designs}"
        ]

        if initial_guess:
            cmd.append("--initial_guess")
        if use_multimer:
            cmd.append("--use_multimer")

        logger.info(f"Running ColabFold command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.work_dir))
            if result.returncode != 0:
                logger.error(f"ColabFold validation failed: {result.stderr}")
                raise RuntimeError(f"ColabFold validation failed: {result.stderr}")
            else:
                logger.info("ColabFold validation completed successfully")

            # Return path to best result
            best_pdb = output_dir / "best.pdb"
            if best_pdb.exists():
                return str(best_pdb)
            else:
                logger.warning("Best PDB not found, returning original structure")
                return f"{output_path}_0.pdb"

        except Exception as e:
            logger.error(f"Error running ColabFold validation: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Local RFdiffusion pipeline with AF2 validation")

    # Required arguments
    parser.add_argument("--rfdiffusion-container", required=True,
                        help="Path to RFdiffusion singularity container")
    parser.add_argument("--colabfold-container", required=True,
                        help="Path to ColabFold singularity container")

    # RFdiffusion arguments
    parser.add_argument("--contigs", default="100",
                        help="Contig specification (default: 100)")
    parser.add_argument("--name", default="test",
                        help="Output name (default: test)")
    parser.add_argument("--pdb", default="",
                        help="Input PDB file or code for conditional generation")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of diffusion iterations (default: 50)")
    parser.add_argument("--num-designs", type=int, default=1,
                        help="Number of designs to generate (default: 1)")
    parser.add_argument("--symmetry", choices=["none", "cyclic", "dihedral"], default="none",
                        help="Symmetry type (default: none)")
    parser.add_argument("--order", type=int, default=1,
                        help="Symmetry order (default: 1)")
    parser.add_argument("--hotspot", default="",
                        help="Hotspot residues for binder design")
    parser.add_argument("--chains", default="",
                        help="Specific chains to use from input PDB")

    # ProteinMPNN/AF2 arguments
    parser.add_argument("--num-seqs", type=int, default=8,
                        help="Number of sequences to generate (default: 8)")
    parser.add_argument("--num-recycles", type=int, default=1,
                        help="Number of AF2 recycles (default: 1)")
    parser.add_argument("--initial-guess", action="store_true",
                        help="Use initial guess for AF2")
    parser.add_argument("--use-multimer", action="store_true",
                        help="Use AF2 multimer")
    parser.add_argument("--rm-aa", default="C",
                        help="Amino acids to remove (default: C)")
    parser.add_argument("--mpnn-sampling-temp", type=float, default=0.1,
                        help="MPNN sampling temperature (default: 0.1)")

    # Other arguments
    parser.add_argument("--work-dir", default="./outputs",
                        help="Working directory (default: ./outputs)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip AF2 validation step")

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = RFdiffusionPipeline(
            singularity_rfdiffusion_path=args.rfdiffusion_container,
            singularity_colabfold_path=args.colabfold_container,
            work_dir=args.work_dir
        )

        # Run RFdiffusion
        logger.info("Starting RFdiffusion...")
        output_path, contigs = pipeline.run_rfdiffusion(
            contigs=args.contigs,
            name=args.name,
            pdb=args.pdb,
            iterations=args.iterations,
            num_designs=args.num_designs,
            symmetry=args.symmetry,
            order=args.order,
            hotspot=args.hotspot,
            chains=args.chains
        )

        logger.info(f"RFdiffusion completed. Output saved to: {output_path}")

        # Run validation if not skipped
        if not args.skip_validation:
            logger.info("Starting ProteinMPNN and AlphaFold2 validation...")
            best_structure = pipeline.run_proteinmpnn_and_alphafold(
                output_path=output_path,
                contigs=contigs,
                num_seqs=args.num_seqs,
                num_recycles=args.num_recycles,
                initial_guess=args.initial_guess,
                use_multimer=args.use_multimer,
                rm_aa=args.rm_aa,
                mpnn_sampling_temp=args.mpnn_sampling_temp,
                num_designs=args.num_designs
            )

            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Best validated structure: {best_structure}")
        else:
            logger.info("Skipping validation step as requested")
            logger.info(f"Generated structure: {output_path}_0.pdb")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()