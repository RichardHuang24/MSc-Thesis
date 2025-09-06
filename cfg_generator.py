# cfg_generator.py
# Manages the Context-Free Grammar for generating and mutating prompts.

import yaml
import random
import re
from typing import Dict, List, Tuple, Optional

class CFGPromptGenerator:
    """
    Handles loading CFG rules, generating random prompt genotypes,
    constructing full prompt text, and mutating genotypes.
    """
    def __init__(self, cfg_rules_path: str):
        with open(cfg_rules_path, 'r') as f:
            self.rules: Dict[str, List[str]] = yaml.safe_load(f)
        self.start_symbol = self.rules['start'][0]
        self.non_terminals = [f"<{key}>" for key in self.rules.keys() if key != 'start']

    def generate_random_genotype(self, baseline: bool = False) -> Dict[str, str]:
        """Generates a random dictionary representing a prompt's genetic makeup.
        If baseline=True, always pick the first (empty) rule for each component."""
        genotype = {}
        structure = self.start_symbol.split(' ')
        for symbol in structure:
            key = symbol.strip('<>')
            if key in self.rules:
                if baseline:
                    # Always pick the first rule (should be empty for ablation)
                    genotype[key] = self.rules[key][0]
                else:
                    genotype[key] = self._expand_symbol(symbol)
        return genotype

    def _expand_symbol(self, symbol: str) -> str:
        """Recursively expands a non-terminal symbol from the grammar."""
        key = symbol.strip('<>')
        if key not in self.rules:
            return symbol # It's a terminal

        # Choose a random production rule for this symbol
        production_rule = random.choice(self.rules[key])
        
        # Find all non-terminals in the chosen rule and expand them
        expanded_parts = []
        parts = re.split('(<[^>]+>)', production_rule)
        for part in parts:
            if part.startswith('<') and part.endswith('>'):
                expanded_parts.append(self._expand_symbol(part))
            else:
                expanded_parts.append(part)
        
        return "".join(expanded_parts)

    def construct_full_prompt(self, genotype: Dict[str, str], problem_text: str) -> str:
        """Constructs the final prompt string from a genotype and a problem."""
        prompt = self.start_symbol
        for key, value in genotype.items():
            # Replace PROBLEM_PLACEHOLDER in gene values as well
            value = value.replace("PROBLEM_PLACEHOLDER", problem_text)
            prompt = prompt.replace(f"<{key}>", value)
        # Replace the final placeholder with the actual problem text (for templates)
        prompt = prompt.replace("<macgyver_problem_text>", problem_text)
        prompt = prompt.replace("<opening_salutation>", "")
        return " ".join(prompt.split()) # Normalize whitespace

    def mutate_genotype(self, genotype: Dict[str, str], mutation_rate: float) -> Dict[str, str]:
        """
        Mutates a genotype by randomly changing some of its genes based on the mutation rate.
        """
        mutated_genotype = genotype.copy()
        for key in mutated_genotype:
            if random.random() < mutation_rate:
                # Re-generate this part of the prompt from the grammar
                mutated_genotype[key] = self._expand_symbol(f"<{key}>")
        return mutated_genotype

    def crossover_genotypes(self, parent1: Dict[str, str], parent2: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Performs single-point crossover between two parent genotypes."""
        child1, child2 = parent1.copy(), parent2.copy()
        keys = list(parent1.keys())
        if len(keys) < 2:
            return child1, child2 # Not enough genes to cross over

        crossover_point = random.randint(1, len(keys) - 1)
        for i in range(crossover_point, len(keys)):
            key = keys[i]
            child1[key], child2[key] = child2[key], child1[key] # Swap genes
            
        return child1, child2
        
    def prompt_text_to_genotype(self, prompt_text: str) -> Optional[Dict[str, str]]:
        """
        A simplified heuristic to reconstruct a genotype from prompt text.
        NOTE: This is non-trivial and may not be perfect if grammar rules overlap.
        This function assumes that each rule choice is unique enough to be identified.
        """
        genotype = {}
        for key, ruleset in self.rules.items():
            if key in ['start', 'macgyver_problem_text']:
                continue
            
            # Find which rule from the ruleset is present in the prompt text
            found_rule = ""
            for rule in ruleset:
                # Clean the rule for matching (remove placeholders)
                clean_rule = re.sub(r'<[^>]+>', '', rule).strip()
                if clean_rule and clean_rule in prompt_text:
                    found_rule = rule
                    break
            genotype[key] = found_rule
            
        # Check if all keys were found
        expected_keys = [k.strip('<>') for k in self.start_symbol.split()]
        if all(k in genotype for k in expected_keys):
            return genotype
        else:
            print(f"Warning: Could not fully reconstruct genotype for prompt: {prompt_text[:100]}...")
            return None


