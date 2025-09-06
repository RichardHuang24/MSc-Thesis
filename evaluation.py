# evaluation.py
# Contains the core logic for evaluating prompts and solutions.

import math
from typing import List, Dict, Any

from llm_services import LLM_API_Handler
import config

class LexicalDiversityCalculator:
    """
    Calculates Mean Segmental Type-Token Ratio (MSTTR) for a list of texts.
    Reference: Malvern, D., & Richards, B. (2002). Investigating Lexical Diversity in Language Learners.
    """
    def __init__(self, segment_length: int = 50):
        self.segment_length = segment_length

    def _segment_text(self, tokens: List[str]) -> List[List[str]]:
        """Splits tokens into segments of fixed length."""
        return [tokens[i:i+self.segment_length] for i in range(0, len(tokens), self.segment_length) if len(tokens[i:i+self.segment_length]) == self.segment_length]

    def msttr(self, texts: List[str]) -> float:
        """
        Calculates the Mean Segmental Type-Token Ratio (MSTTR) for a list of texts.
        Args:
            texts (List[str]): List of text strings (e.g., solution variations).
        Returns:
            float: The mean MSTTR across all texts.
        """
        # --- Lexical diversity calculation (MSTTR) is not used ---
        segment_ratios = []
        for text in texts:
            tokens = text.split()
            segments = self._segment_text(tokens)
            for segment in segments:
                if segment:
                    ttr = len(set(segment)) / len(segment)
                    segment_ratios.append(ttr)
        if not segment_ratios:
            return 0.0
        return sum(segment_ratios) / len(segment_ratios)

class SemanticEntropyCalculator:
    """
    Calculates semantic entropy using an NLI model for semantic clustering.
    """
    def __init__(self, llm_handler: LLM_API_Handler):
        self.llm_handler = llm_handler

    def calculate_entropy(self, solution_variations: List[str]) -> float:
        """Calculates Shannon semantic entropy by clustering variations based on meaning."""
        if not solution_variations or len(solution_variations) < 2:
            return 0.0

        clusters: List[List[str]] = []
        for variation in solution_variations:
            placed = False
            for cluster in clusters:
                score1 = self.llm_handler.check_nli_entailment(premise=cluster[0], hypothesis=variation)
                score2 = self.llm_handler.check_nli_entailment(premise=variation, hypothesis=cluster[0])
                if score1 > config.NLI_ENTAILMENT_THRESHOLD and score2 > config.NLI_ENTAILMENT_THRESHOLD:
                    cluster.append(variation)
                    placed = True
                    break
            if not placed:
                clusters.append([variation])
        
        total_variations = len(solution_variations)
        probabilities = [len(cluster) / total_variations for cluster in clusters]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

class SolutionEvaluator:
    """Orchestrates the full evaluation pipeline for a single prompt."""
    def __init__(self, llm_handler: LLM_API_Handler):
        self.llm_handler = llm_handler
        self.entropy_calculator = SemanticEntropyCalculator(llm_handler)

    def evaluate_prompt(self, full_prompt_text: str, problem_text: str) -> Dict[str, Any]:
        """
        Takes a prompt, generates a full solution, evaluates it, and calculates entropy on its first step.
        """
        # --- NEW, ROBUST WORKFLOW ---
        
        # 1. Generate ONE high-quality solution first.
        main_solution = self.llm_handler.generate_solution(full_prompt_text)
        if not main_solution:
            print("  - FAILED: Generator did not produce a solution.")
            return {
                "raw_divergent": 0.0,  # Renamed from semantic_entropy
                # "msttr": 0.0,  # Commented out lexical diversity
                "raw_convergent": 0.0,
                "safety": 0.0,
                "feasibility": 0.0,
                "effectiveness": 0.0,
                "solution_text": ""
            }

        # 2. Evaluate the complete solution for Convergent Creativity.
        scores = self.llm_handler.evaluate_solution_with_scores(problem_text, main_solution)
        if not scores:
            print("  - FAILED: Evaluator did not produce valid scores.")
            return {
                "raw_divergent": 0.0,  # Renamed from semantic_entropy
                # "msttr": 0.0,  # Commented out lexical diversity
                "raw_convergent": 0.0,
                "safety": 0.0,
                "feasibility": 0.0,
                "effectiveness": 0.0,
                "solution_text": main_solution
            }

        # 3. Generate variations of the FIRST STEP of the good solution for Divergent Creativity.
        variations = self.llm_handler.generate_variations_for_step(main_solution)
        raw_divergent = self.entropy_calculator.calculate_entropy(variations)  # Renamed from semantic_entropy

        # Calculate MSTTR for lexical diversity (now commented out)
        # lex_div_calc = LexicalDiversityCalculator(segment_length=50)
        # msttr = lex_div_calc.msttr(variations)

        # 4. Combine convergent scores using geometric mean.
        s = scores.get("safety", 0.0)
        f = scores.get("feasibility", 0.0)
        e = scores.get("effectiveness", 0.0)
        raw_convergent = (s * f * e) ** (1/3) if all(v > 0 for v in [s, f, e]) else 0.0

        return {
            "raw_divergent": raw_divergent,  # Renamed from semantic_entropy
            # "msttr": msttr,  # Commented out lexical diversity
            "raw_convergent": raw_convergent,
            "safety": s,
            "feasibility": f,
            "effectiveness": e,
            "solution_text": main_solution,
            "scores": [s, f, e]  # <-- Add this line
        }
