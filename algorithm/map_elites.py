# algorithms/map_elites.py
# Implementation of the Multi-Objective MAP-Elites algorithm.

import os
import sys
import json
import random
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

# --- Path Correction ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from .nds_utils import fast_non_dominated_sort, calculate_crowding_distance
from analysis_utils import compute_hypervolume
# from evaluation import LexicalDiversityCalculator  # Commented out, not used

if TYPE_CHECKING:
    from cfg_generator import CFGPromptGenerator
    from task_loader import TaskLoader
    from evaluation import SolutionEvaluator
    from llm_services import LLM_API_Handler
    from umap import UMAP  # type: ignore

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

class MAPElitesAlgorithm:
    def __init__(self, 
                 cfg_generator: 'CFGPromptGenerator', 
                 task_loader: 'TaskLoader', 
                 solution_evaluator: 'SolutionEvaluator', 
                 llm_handler: 'LLM_API_Handler', 
                 umap_model: 'UMAP'):
        self.cfg_generator = cfg_generator
        self.task_loader = task_loader
        self.solution_evaluator = solution_evaluator
        self.llm_handler = llm_handler
        self.umap_model = umap_model
        
        self.grid_shape = config.GRID_SHAPE
        self.archive: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        
        self.umap_min, self.umap_max = self._calculate_umap_bounds()
        print(f"UMAP bounds calculated: Min={self.umap_min}, Max={self.umap_max}")

        self._pending_evals: List[Dict[str, Any]] = []
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        self.fixed_problem_id: Optional[str] = None 
    
    

    def _calculate_umap_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Loads the prompt dataset used for UMAP training to find the projection space boundaries.
        """
        print("Calculating dynamic UMAP boundaries...")
        try:
            with open(config.UMAP_PROMPT_DATASET_PATH, 'r') as f:
                prompts_data = [json.loads(line) for line in f]
            
            prompt_texts = [item['prompt_text'] for item in prompts_data]

            def batch_list(data, batch_size):
                return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

            all_embeddings = []
            batch_size = 32  # Reduce if you still get 413 errors
            for batch in batch_list(prompt_texts, batch_size):
                batch_emb = self.llm_handler.get_embeddings(batch)
                if batch_emb is not None:
                    all_embeddings.append(batch_emb)
            if not all_embeddings:
                raise ValueError("Failed to get embeddings for UMAP bounds calculation.")
            embeddings = np.vstack(all_embeddings)
            
            transformed_coords = self.umap_model.transform(embeddings)
            
            min_bounds = np.min(transformed_coords, axis=0)
            max_bounds = np.max(transformed_coords, axis=0)
            padding = (max_bounds - min_bounds) * 0.10
            
            return (min_bounds - padding).tolist(), (max_bounds + padding).tolist()

        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not calculate dynamic UMAP bounds ({e}). Falling back to default [-5, 5].")
            return [-5.0, -5.0], [5.0, 5.0]

    def _get_bin_coords(self, bd_float_coords: np.ndarray) -> Tuple[int, int]:
        """Maps continuous BD coordinates to discrete grid bin indices."""
        x, y = bd_float_coords
        x_bin = int(np.clip((x - self.umap_min[0]) / (self.umap_max[0] - self.umap_min[0]) * self.grid_shape[0], 0, self.grid_shape[0] - 1))
        y_bin = int(np.clip((y - self.umap_min[1]) / (self.umap_max[1] - self.umap_min[1]) * self.grid_shape[1], 0, self.grid_shape[1] - 1))
        return (x_bin, y_bin)

    def _place_in_archive(self, candidate: Dict[str, Any]):
        """Places a new candidate into the archive using NDS for its cell."""
        bin_coords = candidate['bd_bin_coords']
        current_front = self.archive.get(bin_coords, [])
        combined = current_front + [candidate]
        
        front_indices_list = fast_non_dominated_sort(combined)
        if not front_indices_list: 
            return
        
        new_front_indices = front_indices_list[0]
        new_front = [combined[i] for i in new_front_indices]

        if len(new_front) > config.CELL_CAPACITY_LIMIT:
            distances = calculate_crowding_distance(new_front)
            sorted_by_crowding = sorted(zip(new_front, distances), key=lambda x: x[1], reverse=True)
            new_front = [item for item, dist in sorted_by_crowding[:config.CELL_CAPACITY_LIMIT]]

        self.archive[bin_coords] = new_front

    def _evaluate_and_place_new_individual(self, generation: int, genotype: Optional[Dict] = None, parent_prompt_text: Optional[str] = None):
        if genotype is None:
            genotype = self.cfg_generator.generate_random_genotype()
        
        if self.fixed_problem_id is not None:
            problem = self.task_loader.get_problem_by_id(self.fixed_problem_id)
        else:
            problem = self.task_loader.get_random_problem()
            
        genotype = genotype.copy()
        genotype["macgyver_problem_text"] = problem['problem_text']  # Store actual problem text
        prompt_text = self.cfg_generator.construct_full_prompt(genotype, problem['problem_text'])

        eval_results = self.solution_evaluator.evaluate_prompt(prompt_text, problem['problem_text'])

        # --- Embed only the prompt style (without the problem text) for BD ---
        # Construct a style-only string (adjust keys as needed for your genes)
        style_string = f"{genotype.get('role_instruction', '')} {genotype.get('format_instruction', '')} {genotype.get('creativity_instruction', '')}".strip()

        if style_string in self._embedding_cache:
            embedding = self._embedding_cache[style_string]
        else:
            embedding_result = self.llm_handler.get_embeddings([style_string])
            if embedding_result is not None:
                embedding = embedding_result
            else:
                return
            if embedding is not None:
                self._embedding_cache[style_string] = embedding
        if embedding is None: 
            return

        bd_float = self.umap_model.transform(embedding.reshape(1, -1))[0]
        bd_bin = self._get_bin_coords(bd_float)

        pure_solution = extract_pure_solution(eval_results['solution_text'])  # Only assistant reply

        #individual_id = hashlib.md5((json.dumps(genotype, sort_keys=True) + str(problem['id'])).encode()).hexdigest()

        candidate_data = {
            "genotype": genotype,
            "parent_prompt_text": parent_prompt_text,  # Store parent prompt string
            "prompt_text": prompt_text,
            "scores": [eval_results['raw_divergent'], eval_results['raw_convergent']],
            "individual_scores": eval_results['scores'],
            "solution_text": pure_solution,  # Only the assistant's reply
            "problem_id": problem['problem_id'],
            "category": problem['category'],  # Store problem category
            "generation": generation,
            "bd_float_coords": bd_float.tolist(),
            "bd_bin_coords": bd_bin
        }
        
        self._place_in_archive(candidate_data)
        self._pending_evals.append(candidate_data)  # Buffer instead of writing

    def _log_generation_summary(self, generation: int):
        """Calculates and saves summary statistics for the current generation."""
        if not self.archive: 
            return
        all_elites = [elite for cell in self.archive.values() for elite in cell]
        df = pd.DataFrame(all_elites)
        df[['raw_divergent', 'raw_convergent']] = pd.DataFrame(df['scores'].tolist(), index=df.index)
        
        # Calculate MSTTR for all solutions in the archive
        # lex_div_calc = LexicalDiversityCalculator(segment_length=5)
        # msttr = lex_div_calc.msttr(df['solution_text'].tolist()) if not df.empty else 0.0

        # Calculate average divergent creativity (semantic entropy)
        avg_divergent_creativity = df['raw_divergent'].mean() if not df.empty else 0.0
        max_divergent_creativity = df['raw_divergent'].max() if not df.empty else 0.0

        # Score distributions
        divergent_scores = df['raw_divergent'].tolist() if not df.empty else []
        convergent_scores = df['raw_convergent'].tolist() if not df.empty else []

        # Parentage/lineage tracking
        parent_ids = df['parent_id'].tolist() if 'parent_id' in df.columns else []

        log_entry = {
            "generation": generation,
            "num_cells": len(self.archive),
            "total_elites": len(all_elites),
            "avg_divergent_creativity": avg_divergent_creativity,  # Renamed from avg_semantic_entropy
            "max_divergent_creativity": max_divergent_creativity,
            "avg_convergent": df['raw_convergent'].mean(),
            "max_convergent": df['raw_convergent'].max(),
            # "msttr": msttr,  # Commented out lexical diversity
            "divergent_scores": divergent_scores,  # Score distribution
            "convergent_scores": convergent_scores,  # Score distribution
            "parent_ids": parent_ids  # Parent tracking
        }
        with open(os.path.join(config.RESULTS_DIR, "map_elites_evolution_log.jsonl"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"Generation {generation}: {log_entry['num_cells']} cells, {log_entry['total_elites']} elites. Avg Conv: {log_entry['avg_convergent']:.2f}, Avg Divergent: {avg_divergent_creativity:.4f}")

        # Compute hypervolume and archive convergence
        points = df[['raw_divergent', 'raw_convergent']].values.tolist()
        points = [[max(0.0, x), max(0.0, y)] for x, y in points]
        if points:
            max_div = max(x for x, y in points)
            max_conv = max(y for x, y in points)
            reference_point = [max_div + 1.0, max_conv + 1.0]
        else:
            reference_point = [1.0, 1.0]
        hypervolume = compute_hypervolume(points, reference_point)
        archive_size = len(all_elites)
        hv_log = {
            "generation": generation,
            "hypervolume": hypervolume,
            "archive_size": archive_size
        }
        with open(os.path.join(config.RESULTS_DIR, "map_elites_hypervolume_log.jsonl"), "a") as f:
            f.write(json.dumps(hv_log) + "\n")

        # Write buffered results
        if self._pending_evals:
            with open(os.path.join(config.RESULTS_DIR, "map_elites_evaluated_prompts.jsonl"), "a") as f:
                for item in self._pending_evals:
                    f.write(json.dumps(item) + "\n")
            self._pending_evals = []

    def run(self):
        """Executes the main MAP-Elites evolutionary loop."""
        print("--- Starting MAP-Elites Run ---")
        
        # Clear previous log files
        open(os.path.join(config.RESULTS_DIR, "map_elites_evaluated_prompts.jsonl"), "w").close()
        open(os.path.join(config.RESULTS_DIR, "map_elites_evolution_log.jsonl"), "w").close()

        print(f"Generating {config.NUM_INITIAL_POPULATION} initial prompts...")
        # First individual: empty/baseline
        self._evaluate_and_place_new_individual(
            generation=0, 
            genotype=self.cfg_generator.generate_random_genotype(baseline=True),
            parent_prompt_text=None
        )
        # The rest: random genotypes
        for _ in tqdm(range(config.NUM_INITIAL_POPULATION - 1), desc="Initial Population (random)"):
            self._evaluate_and_place_new_individual(
                generation=0, 
                genotype=self.cfg_generator.generate_random_genotype(),
                parent_prompt_text=None
            )
        self._log_generation_summary(0)
        self.save_archive("map_elites_archive_gen_0.json")  # Save initial archive

        for gen in range(1, config.NUM_GENERATIONS + 1):
            if not self.archive:
                print("Archive is empty. Cannot select parents. Generating a random individual.")
                self._evaluate_and_place_new_individual(generation=gen)
                self._log_generation_summary(gen)
                if gen % 5 == 0:
                    self.save_archive(f"map_elites_archive_gen_{gen}.json")
                continue

            print(f"\n--- Generation {gen}/{config.NUM_GENERATIONS} ---")
            for _ in tqdm(range(config.POPULATION_SIZE), desc=f"Generation {gen}"):
                parent_bin = random.choice(list(self.archive.keys()))
                parent = random.choice(self.archive[parent_bin])
                # parent_id = parent.get("id", None)  # This is correct, just ensure parent['id'] is always set
                mutated_genotype = self.cfg_generator.mutate_genotype(parent['genotype'], config.MUTATION_RATE)
                parent_prompt_text = parent.get("prompt_text", None)
                self._evaluate_and_place_new_individual(
                    generation=gen, 
                    genotype=mutated_genotype,
                    parent_prompt_text=parent_prompt_text
                )
            
            self._log_generation_summary(gen)
            if gen % 5 == 0:
                self.save_archive(f"map_elites_archive_gen_{gen}.json")

        print("--- MAP-Elites Run Finished ---")
        self.save_archive("map_elites_final_archive.json")
        return self.archive
        
    def save_archive(self, filename: str):
        serializable_archive = {str(k): v for k, v in self.archive.items()}
        with open(os.path.join(config.RESULTS_DIR, filename), 'w') as f:
            json.dump(serializable_archive, f, indent=2)
        print(f"Archive saved to {filename}")
