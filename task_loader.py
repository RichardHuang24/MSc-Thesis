# task_loader.py
# Loads and manages the MacGyver dataset.

import json
import random
from typing import List, Dict, Any

class TaskLoader:
    """
    Loads the MacGyver dataset from a JSON file and provides random problems.
    """
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path (str): Path to the macgyver_dataset.json file.
        """
        try:
            with open(dataset_path, 'r') as f:
                self.problems: List[Dict[str, Any]] = json.load(f)
            print(f"Successfully loaded {len(self.problems)} problems from {dataset_path}.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading dataset from {dataset_path}: {e}")
            self.problems = []
            
    def get_problem_by_id(self, problem_id: str):
        """
        Returns the problem dict with the given id, or raises ValueError if not found.
        """
        for problem in self.problems:
            if problem["problem_id"] == problem_id:
                return problem
        raise ValueError(f"Problem with id '{problem_id}' not found.")
    
    def get_random_problem(self) -> Dict[str, Any]:
        """
        Returns a random problem dictionary from the loaded dataset.
        Each problem dict contains 'id', 'problem_text', and 'ground_truth_solution'.
        """
        if not self.problems:
            # Return a default problem if the dataset failed to load
            return {
                "id": "default_problem",
                "problem_text": "You are trapped in a room with a locked door. You have a paperclip and a piece of string. How do you escape?",
                "ground_truth_solution": "Fashion a lockpick from the paperclip."
            }
        return random.choice(self.problems)

