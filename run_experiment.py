# run_experiment.py
# A master script to orchestrate the entire thesis experiment workflow.

import subprocess
import argparse
import sys
import os

# Ensure the script can find other modules in the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_command(command: list, description: str):
    """Helper function to run a command and print its status."""
    print(f"\n{'='*20}\n[RUNNING] {description}\n{'='*20}")
    try:
        # Using sys.executable ensures we use the same python interpreter
        process = subprocess.run([sys.executable, "-u"] + command, check=True, text=True)
        print(f"[SUCCESS] {description} completed.")
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description} failed with exit code {e.returncode}.")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        sys.exit(1) # Exit if a step fails
    except FileNotFoundError:
        print(f"[FAILED] Command not found: {command[0]}. Make sure the script exists.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Master runner for the computational creativity thesis experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=["prepare-umap", "run-ga", "run-me", "analyze", "all"],
        help="""The stage of the experiment to run:
  - prepare-umap: Generates prompt dataset and trains the UMAP model.
  - run-ga:       Runs the NSGA-II genetic algorithm experiment.
  - run-me:       Runs the MAP-Elites experiment.
  - analyze:      Runs the post-experiment analysis and generates plots.
  - all:          Runs all stages sequentially (prepare, run-ga, run-me, analyze).
"""
    )
    parser.add_argument(
        "--problem-id",
        type=str,
        default=None,
        help="If set, use this problem id for all individuals (fixed problem mode)."
    )
    args = parser.parse_args()

    stage = args.stage

    if stage == "prepare-umap" or stage == "all":
        run_command(["prepare_umap.py"], "UMAP Preparation")

    if stage == "run-ga" or stage == "all":
        run_command(["main.py", "ga"], "Genetic Algorithm (NSGA-II) Run")

    if stage == "run-me" or stage == "all":
        cmd = ["main.py", "map_elites"]
        if args.problem_id:
            cmd += ["--problem-id", args.problem_id]
        run_command(cmd, "MAP-Elites Run")

    if stage == "analyze" or stage == "all":
        run_command(["analysis_utils.py"], "Post-Experiment Analysis")

    print("\nWorkflow finished.")

if __name__ == "__main__":
    main()

