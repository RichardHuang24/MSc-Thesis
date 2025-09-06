# algorithms/genetic_algorithm.py
# Implementation of the NSGA-II Genetic Algorithm.

import os
import sys
import json
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import re

# --- Path Correction ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from .nds_utils import fast_non_dominated_sort, calculate_crowding_distance
from analysis_utils import compute_hypervolume

if TYPE_CHECKING:
    from cfg_generator import CFGPromptGenerator
    from task_loader import TaskLoader
    from evaluation import SolutionEvaluator

def extract_pure_solution(raw_output: str) -> str:
    """
    Extracts only the assistant's reply from a raw LLM output with role markers.
    """
    # Try to extract only the assistant's reply
    match = re.search(r"<\|im_start\|>assistant\n(.*?)(?=<\|im_end\|>|$)", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: remove everything before the first assistant marker
    assistant_start = raw_output.find("<|im_start|>assistant")
    if assistant_start != -1:
        after = raw_output[assistant_start:]
        after = re.sub(r"<\|im_start\|>assistant\n?", "", after)
        after = re.sub(r"<\|im_end\|>", "", after)
        return after.strip()
    return raw_output.strip()

def decode_unicode(text):
    if isinstance(text, str):
        return text.encode('utf-8').decode('unicode_escape')
    return text

class GeneticAlgorithm:
    def __init__(self, 
                 cfg_generator: 'CFGPromptGenerator', 
                 task_loader: 'TaskLoader', 
                 solution_evaluator: 'SolutionEvaluator'):
        self.cfg_generator = cfg_generator
        self.task_loader = task_loader
        self.solution_evaluator = solution_evaluator
        self.population: List[Dict[str, Any]] = []
        self.fixed_problem_id: Optional[str] = None  # <-- Add this line

    def _create_and_evaluate_individual(self, generation: int, genotype: Optional[Dict] = None) -> Dict[str, Any]:
        if genotype is None:
            genotype = self.cfg_generator.generate_random_genotype()
        # Use fixed problem if set
        if self.fixed_problem_id is not None:
            problem = self.task_loader.get_problem_by_id(self.fixed_problem_id)
        else:
            problem = self.task_loader.get_random_problem()
        genotype = genotype.copy()
        if "start" in genotype:
            del genotype["start"]
        genotype["macgyver_problem_text"] = problem['problem_text']
        prompt_text = self.cfg_generator.construct_full_prompt(genotype, problem['problem_text'])
        eval_results = self.solution_evaluator.evaluate_prompt(prompt_text, problem['problem_text'])
        prompt_text = decode_unicode(prompt_text)
        solution_text = extract_pure_solution(decode_unicode(eval_results['solution_text']))
        candidate_data = {
            "genotype": genotype,
            "prompt_text": prompt_text,
            "scores": [eval_results['raw_divergent'], eval_results['raw_convergent']],
            "individual_scores": eval_results['scores'],
            "solution_text": solution_text,
            "problem_id": problem['id'],
            "category": problem['category'],  # <-- Add this line
            "generation": generation
        }
        return candidate_data

    def _select_parents_tournament(self) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Selects parents using a more efficient tournament selection.
        """
        parents = []
        if not self.population: 
            return []
        
        for _ in range(config.POPULATION_SIZE):
            p1, p2 = random.sample(self.population, 2)
            
            # This simple tournament selection is faster than full ranking for large populations
            p1_scores = p1['scores']
            p2_scores = p2['scores']

            if (p1_scores[0] > p2_scores[0] and p1_scores[1] > p2_scores[1]):
                parents.append(p1)
            elif (p2_scores[0] > p1_scores[0] and p2_scores[1] > p1_scores[1]):
                parents.append(p2)
            else: # If non-dominated, choose one randomly
                parents.append(random.choice([p1, p2]))
        return parents

    def _log_generation_summary(self, generation: int):
        """Calculates and saves summary statistics for the current generation."""
        if not self.population: 
            return
        df = pd.DataFrame(self.population)
        df[['raw_divergent', 'raw_convergent']] = pd.DataFrame(df['scores'].tolist(), index=df.index)
        fronts = fast_non_dominated_sort(self.population)
        num_non_dominated = len(fronts[0]) if fronts else 0

        # Score distributions
        divergent_scores = df['raw_divergent'].tolist() if not df.empty else []
        convergent_scores = df['raw_convergent'].tolist() if not df.empty else []

        log_entry = {
            "generation": generation,
            "population_size": len(self.population),
            "num_non_dominated": num_non_dominated,
            "avg_divergent_creativity": df['raw_divergent'].mean(),
            "max_divergent_creativity": df['raw_divergent'].max(),
            "avg_convergent": df['raw_convergent'].mean(),
            "max_convergent": df['raw_convergent'].max(),
            "divergent_scores": divergent_scores,
            "convergent_scores": convergent_scores,
            # Optionally add parent_prompt_text if you track it
        }
        with open(os.path.join(config.RESULTS_DIR, "ga_evolution_log.jsonl"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"Generation {generation}: Pop Size={log_entry['population_size']}, Non-Dominated={log_entry['num_non_dominated']}. Avg Conv: {log_entry['avg_convergent']:.2f}, Avg Divergent: {log_entry['avg_divergent_creativity']:.4f}")

        # Compute hypervolume and archive convergence
        points = df[['raw_divergent', 'raw_convergent']].values.tolist()
        points = [[max(0.0, x), max(0.0, y)] for x, y in points]  # Clamp negatives to zero
        reference_point = [0.0, 0.0]
        hypervolume = compute_hypervolume(points, reference_point)
        hv_log = {
            "generation": generation,
            "hypervolume": hypervolume
        }
        with open(os.path.join(config.RESULTS_DIR, "ga_hypervolume_log.jsonl"), "a") as f:
            f.write(json.dumps(hv_log) + "\n")

    def get_empty_genotype(self):
        """
        Constructs a genotype using the first entry of each grammar component.
        """
        # Use the first entry for each grammar key
        return {k: v[0] for k, v in self.cfg_generator.rules.items()}

    def run(self):
        """Executes the main NSGA-II evolutionary loop."""
        print("--- Starting Genetic Algorithm (NSGA-II) Run ---")

        # Clear previous log files
        open(os.path.join(config.RESULTS_DIR, "ga_evaluated_prompts.jsonl"), "w").close()
        open(os.path.join(config.RESULTS_DIR, "ga_evolution_log.jsonl"), "w").close()
        open(os.path.join(config.RESULTS_DIR, "ga_hypervolume_log.jsonl"), "w").close()

        print(f"Generating and evaluating {config.POPULATION_SIZE} initial individuals...")
        empty_genotype = self.get_empty_genotype()
        # First individual: empty/baseline
        individual = self._create_and_evaluate_individual(generation=0, genotype=empty_genotype.copy())
        self.population.append(individual)
        # The rest: random genotypes
        for _ in tqdm(range(config.POPULATION_SIZE - 1), desc="Initial Population (random)"):
            random_genotype = self.cfg_generator.generate_random_genotype()
            individual = self._create_and_evaluate_individual(generation=0, genotype=random_genotype)
            self.population.append(individual)
        self._log_generation_summary(0)
        self.save_population("ga_population_gen_0.json")  # Save initial population

        for gen in range(1, config.NUM_GENERATIONS + 1):
            print(f"\n--- Generation {gen}/{config.NUM_GENERATIONS} ---")
            parents = self._select_parents_tournament()
            offspring = []

            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[i+1] if (i+1) < len(parents) else parents[0]
                if random.random() < config.CROSSOVER_RATE:
                    c1_geno, c2_geno = self.cfg_generator.crossover_genotypes(p1['genotype'], p2['genotype'])
                else:
                    c1_geno, c2_geno = p1['genotype'], p2['genotype']
                c1_geno = self.cfg_generator.mutate_genotype(c1_geno, config.MUTATION_RATE)
                c2_geno = self.cfg_generator.mutate_genotype(c2_geno, config.MUTATION_RATE)
                offspring.append(self._create_and_evaluate_individual(gen, c1_geno))
                offspring.append(self._create_and_evaluate_individual(gen, c2_geno))

            combined_population = self.population + offspring
            fronts = fast_non_dominated_sort(combined_population)
            next_population = []
            for front_indices in fronts:
                if not front_indices: 
                    continue
                if len(next_population) + len(front_indices) <= config.POPULATION_SIZE:
                    next_population.extend([combined_population[i] for i in front_indices])
                else:
                    front_individuals = [combined_population[i] for i in front_indices]
                    distances = calculate_crowding_distance(front_individuals)
                    sorted_front = sorted(zip(front_individuals, distances), key=lambda x: x[1], reverse=True)
                    remaining_slots = config.POPULATION_SIZE - len(next_population)
                    next_population.extend([ind for ind, dist in sorted_front[:remaining_slots]])
                    break

            self.population = next_population
            self._log_generation_summary(gen)
            if gen % 2 == 0:
                self.save_population(f"ga_population_gen_{gen}.json")  # Save every 2 generations

        print("--- Genetic Algorithm Run Finished ---")
        self.save_population("ga_final_population.json")
        return self.population

    def save_population(self, filename: str):
        with open(os.path.join(config.RESULTS_DIR, filename), 'w') as f:
            json.dump(self.population, f, indent=2)
        print(f"Population saved to {filename}")
