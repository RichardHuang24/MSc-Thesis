# main.py
# Main script to run the evolutionary experiments for the thesis.

import os
import pickle
import argparse
import sys
from algorithm.baseline_no_prompt import run_baseline_no_prompt
from algorithm.baseline_random_prompt import run_baseline_random_prompt

# Ensure the script can find other modules in the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from llm_services import LLM_API_Handler
from cfg_generator import CFGPromptGenerator
from task_loader import TaskLoader
from evaluation import SolutionEvaluator
from algorithm.map_elites import MAPElitesAlgorithm
from algorithm.genetic_algorithm import GeneticAlgorithm

def main():
    parser = argparse.ArgumentParser(description="Run evolutionary prompt engineering experiments.")
    parser.add_argument(
        "algorithm", 
        choices=["map_elites", "ga", "baseline_no_prompt", "baseline_random_prompt"],
        help="The algorithm to run: 'map_elites', 'ga', 'baseline_no_prompt', or 'baseline_random_prompt'."
    )
    parser.add_argument(
        "--problem-id", 
        type=str, 
        default=None, 
        help="Run with a fixed problem id"
    )
    args = parser.parse_args()

    # --- 1. Initialization & Prerequisite Checks (with loud failures) ---
    print("--- Initializing Framework Components ---")
    
    # Check for API key (FIXED: Made the check more specific)
    placeholder_keys = ["YOUR_HUGGINGFACE_API_KEY_HERE", "XYZ"]
    if config.HF_API_KEY in placeholder_keys:
        # Raise an error instead of exiting silently
        raise ValueError("Prerequisite Check FAILED: Please set your real Hugging Face API key in config.py")
    
    # Check for UMAP model (required by MAP-Elites)
    if args.algorithm == "map_elites" and not os.path.exists(config.UMAP_MODEL_PATH):
        # Raise an error instead of exiting silently
        raise FileNotFoundError(f"Prerequisite Check FAILED: UMAP model not found at {config.UMAP_MODEL_PATH}. Please run 'python run_experiment.py prepare-umap' first.")

    # Load components
    llm_handler = LLM_API_Handler(config.HF_API_KEY)
    cfg_generator = CFGPromptGenerator(config.CFG_RULES_PATH)
    task_loader = TaskLoader(config.MACGYVER_DATASET_PATH)
    solution_evaluator = SolutionEvaluator(llm_handler)
    
    print("Components initialized successfully.")

    # --- 2. Run Selected Algorithm ---
    if args.algorithm == "map_elites":
        with open(config.UMAP_MODEL_PATH, 'rb') as f:
            umap_model = pickle.load(f)
        
        algorithm = MAPElitesAlgorithm(
            cfg_generator=cfg_generator,
            task_loader=task_loader,
            solution_evaluator=solution_evaluator,
            llm_handler=llm_handler,
            umap_model=umap_model
        )
        if args.problem_id:
            algorithm.fixed_problem_id = args.problem_id
        algorithm.run()

    elif args.algorithm == "ga":
        algorithm = GeneticAlgorithm(
            cfg_generator=cfg_generator,
            task_loader=task_loader,
            solution_evaluator=solution_evaluator
        )
        if args.problem_id:
            algorithm.fixed_problem_id = args.problem_id
        algorithm.run()

    elif args.algorithm == "baseline_no_prompt":
        print("Running Baseline: No Prompt")
        run_baseline_no_prompt(num_runs=20, problem_id=args.problem_id)
        return

    elif args.algorithm == "baseline_random_prompt":
        print("Running Baseline: Random Prompt")
        run_baseline_random_prompt(num_runs=20, problem_id=args.problem_id)
        return

    print("\n--- Experiment Finished ---")
    print(f"Results have been saved to the '{config.RESULTS_DIR}' directory.")
    print("You can now run 'python analysis.py' to generate plots.")

    # Run Baseline Experiments
    print("Running Baseline: No Prompt")
    run_baseline_no_prompt(num_runs=20)
    print("Running Baseline: Random Prompt")
    run_baseline_random_prompt(num_runs=20)
    # ... any post-processing or analysis ...


if __name__ == "__main__":
    main()
