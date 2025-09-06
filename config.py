# config.py
# Central configuration file for the thesis experiment framework.

import os

# --- API and Model Endpoint Configuration ---
# Your Hugging Face API key. It's recommended to set this as an environment variable.
HF_API_KEY = "" # **IMPORTANT: Replace or set as env var**

# --- Hugging Face Inference Endpoint URLs ---
# Direct URLs to your deployed private endpoints.
# The model assignments are based on best practices for this experiment.

# Generator: Llama 3 is a powerful model for creative solution generation.
GENERATOR_LLM_ENDPOINT = "https://nicac9bnl723acty.us-east-1.aws.endpoints.huggingface.cloud" 

# Evaluator: Qwen provides an independent assessment, reducing bias.
EVALUATOR_LLM_ENDPOINT = "https://us6qw3kufu6n0rwi.us-east-1.aws.endpoints.huggingface.cloud" 

# Embedding: all-MiniLM is the standard for efficient and effective sentence embeddings.
EMBEDDING_LLM_ENDPOINT = "https://o3jty2wadjfqgs48.us-east-1.aws.endpoints.huggingface.cloud" 

# NLI: A specialized public model is used for accurate semantic clustering.
# This is generally more reliable than using a general-purpose model for this specific task.
NLI_MODEL_ENDPOINT = "https://o61g3q0p0o7796bl.us-east-1.aws.endpoints.huggingface.cloud"


# --- File and Directory Paths ---
DATA_DIR = "data"
RESULTS_DIR = "results"
ANALYSIS_DIR = "analysis_plots"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Input data files
CFG_RULES_PATH = os.path.join(DATA_DIR, "cfg_rules.yaml")
MACGYVER_DATASET_PATH = os.path.join(DATA_DIR, "macgyver_dataset.json")

# Output files from UMAP preparation
UMAP_MODEL_PATH = os.path.join(RESULTS_DIR, "trained_umap_model.pkl")
UMAP_PROMPT_DATASET_PATH = os.path.join(RESULTS_DIR, "umap_prompt_dataset.jsonl")


# --- Semantic Entropy Calculation ---
# Number of variations to generate for the entropy calculation.
NUM_SOLUTION_VARIATIONS = 5
# NLI entailment threshold for clustering. A higher value means stricter clustering.
NLI_ENTAILMENT_THRESHOLD = 0.6


# --- UMAP Configuration (for Behavior Descriptors) ---
# Number of prompts to generate for training the UMAP model.
UMAP_TRAINING_PROMPTS = 1500
UMAP_N_NEIGHBORS = 15  # Default is 15. Controls local vs. global structure.
UMAP_MIN_DIST = 0.1   # Default is 0.1. Controls how tightly points are packed.
UMAP_N_COMPONENTS = 2 # 2 for 2D visualization.


# --- Evolutionary Algorithm Hyperparameters ---
# NOTE: The framework uses a multi-objective approach (Non-Dominated Sorting), 
# which is more powerful than a single weighted-sum fitness. Therefore, 
# weights like W_DIVERGENT are not needed. The algorithms will find the
# optimal trade-off front between the two objectives automatically.

# General settings
NUM_GENERATIONS =  20 # Number of generations to run the main loop.
NUM_INITIAL_POPULATION = 8 # Number of initial random individuals.

# MAP-Elites specific settings
GRID_SHAPE = (10, 10) # The resolution of the MAP-Elites grid.
CELL_CAPACITY_LIMIT = 5 # Max number of elites on a cell's Pareto front.

# Genetic Algorithm (NSGA-II) specific settings
POPULATION_SIZE = 5
# The probability of mutating a single gene in a genotype.
MUTATION_RATE = 0.5
# The probability of performing crossover between two parents.
CROSSOVER_RATE = 0.5
