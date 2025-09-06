# algorithms/nds_utils.py
# Utility functions for Non-Dominated Sorting (NDS) and crowding distance,
# which are core components of NSGA-II and multi-objective MAP-Elites.

import numpy as np
from typing import List, Dict, Any

def fast_non_dominated_sort(population: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Performs fast non-dominated sorting on a population.
    Args:
        population: A list of individuals, where each individual is a dict
                    that must contain a 'scores' key with a list/tuple of objective values.
    Returns:
        A list of fronts, where each front is a list of indices of individuals.
    """
    if not population:
        return []

    values = np.array([ind['scores'] for ind in population])
    n = len(population)
    
    dominating_counts = np.zeros(n, dtype=int)
    dominated_sets = [[] for _ in range(n)]
    
    # --- FIXED: Corrected initialization of fronts ---
    # Start with an empty list for the first front. It will be populated below.
    fronts = [[]] 

    for i in range(n):
        for j in range(i + 1, n):
            # Check for domination
            if np.all(values[i] >= values[j]) and np.any(values[i] > values[j]): # i dominates j
                dominating_counts[j] += 1
                dominated_sets[i].append(j)
            elif np.all(values[j] >= values[i]) and np.any(values[j] > values[i]): # j dominates i
                dominating_counts[i] += 1
                dominated_sets[j].append(i)

    # Find the first front (all individuals with domination count of 0)
    for i in range(n):
        if dominating_counts[i] == 0:
            fronts[0].append(i)
    
    # Build subsequent fronts
    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_sets[i]:
                dominating_counts[j] -= 1
                if dominating_counts[j] == 0:
                    next_front.append(j)
        
        if not next_front:
            break # No more fronts to create
            
        fronts.append(next_front)
        front_idx += 1
            
    return fronts

def calculate_crowding_distance(front: List[Dict[str, Any]]) -> List[float]:
    """
    Calculates the crowding distance for each individual in a single front.
    """
    n = len(front)
    if n <= 2: # Boundary points get infinite distance, no need to calculate for 0, 1, or 2 individuals
        return [np.inf] * n

    distances = np.zeros(n)
    scores = np.array([ind['scores'] for ind in front])
    num_objectives = scores.shape[1]

    for m in range(num_objectives):
        sorted_indices = np.argsort(scores[:, m])
        
        # Assign infinite distance to boundary points
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        min_val = scores[sorted_indices[0], m]
        max_val = scores[sorted_indices[-1], m]
        
        if max_val == min_val:
            continue

        # Calculate distance for intermediate points
        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (scores[sorted_indices[i+1], m] - scores[sorted_indices[i-1], m]) / (max_val - min_val)

    return distances.tolist()

