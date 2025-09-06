import os
import json
from tqdm import tqdm
import random
import numpy as np

import config
from task_loader import TaskLoader
from evaluation import SolutionEvaluator
from llm_services import LLM_API_Handler

random.seed(42)
np.random.seed(42)

RESULTS_PATH = os.path.join(config.RESULTS_DIR, "baseline_no_prompt_results.jsonl")

def extract_pure_problem(problem_text: str) -> str:
    return problem_text.strip()

def extract_pure_solution(solution_text: str) -> str:
    # Use the same logic as in map_elites.py
    import re
    match = re.search(r"<\|im_start\|>assistant\n(.*?)(?=<\|im_end\|>|$)", solution_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    assistant_start = solution_text.find("<|im_start|>assistant")
    if assistant_start != -1:
        after = solution_text[assistant_start:]
        after = re.sub(r"<\|im_start\|>assistant\n?", "", after)
        after = re.sub(r"<\|im_end\|>", "", after)
        return after.strip()
    return solution_text.strip()

def run_baseline_no_prompt(num_runs=20, problem_id=None):
    llm_handler = LLM_API_Handler(config.HF_API_KEY)
    task_loader = TaskLoader(config.MACGYVER_DATASET_PATH)
    solution_evaluator = SolutionEvaluator(llm_handler)
    if problem_id is not None:
        problems = [task_loader.get_problem_by_id(str(problem_id))]
    else:
        problems = task_loader.problems
    all_results = []

    for run in tqdm(range(num_runs), desc="Baseline No Prompt Runs"):
        for problem in problems:
            pure_problem_text = extract_pure_problem(problem['problem_text'])
            eval_results = solution_evaluator.evaluate_prompt(pure_problem_text, pure_problem_text)
            pure_solution = extract_pure_solution(eval_results["solution_text"])
            result = {
                "run": run,
                "problem_id": problem['problem_id'],
                "category": problem['category'],
                "problem_text": pure_problem_text,
                "solution_text": pure_solution,
                "raw_convergent": eval_results['raw_convergent'],
                "raw_divergent": eval_results['raw_divergent'],
                "scores": eval_results['scores']
            }
            all_results.append(result)
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(result) + "\n")
    print(f"Baseline no-prompt results saved to {RESULTS_PATH}")
    return all_results

if __name__ == "__main__":
    run_baseline_no_prompt(num_runs=20)