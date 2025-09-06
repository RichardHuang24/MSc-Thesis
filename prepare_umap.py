# prepare_umap.py
# A one-time script to generate a dataset of prompts and train a UMAP model.
# This ensures a consistent projection for all subsequent algorithm runs.

import json
import pickle
import numpy as np
from umap import UMAP # type: ignore
from tqdm import tqdm
from typing import List

import config
from cfg_generator import CFGPromptGenerator
from task_loader import TaskLoader
from llm_services import LLM_API_Handler

def batch_list(data: List, batch_size: int) -> List[List]:
    """Splits a list into smaller chunks of a specified size."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def prepare_umap_model():
    """
    Generates a large set of diverse prompts, embeds them in batches, 
    and trains a UMAP model.
    """
    print("--- Starting UMAP Model Preparation ---")
    
    # 1. Initialize components
    cfg_generator = CFGPromptGenerator(config.CFG_RULES_PATH)
    task_loader = TaskLoader(config.MACGYVER_DATASET_PATH)
    llm_handler = LLM_API_Handler(config.HF_API_KEY)

    # 2. Generate a large, diverse dataset of sample prompts
    print(f"Generating {config.UMAP_TRAINING_PROMPTS} sample prompts...")
    prompts_data = []
    for _ in tqdm(range(config.UMAP_TRAINING_PROMPTS), desc="Generating Prompts"):
        genotype = cfg_generator.generate_random_genotype()
        problem = task_loader.get_random_problem()
        prompt_text = cfg_generator.construct_full_prompt(genotype, problem['problem_text'])
        prompts_data.append({"genotype": genotype, "prompt_text": prompt_text})

    # Save the generated prompts for inspection
    with open(config.UMAP_PROMPT_DATASET_PATH, 'w') as f:
        for item in prompts_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved prompt dataset to {config.UMAP_PROMPT_DATASET_PATH}")

    # 3. Embed the sample prompts in batches
    print("Embedding prompts in batches... (This may take a while)")
    prompt_texts = [item['prompt_text'] for item in prompts_data]
    
    # --- FIXED: Batching logic to handle API limits ---
    batch_size = 32  # Set your model's batch size limit here
    all_embeddings = []
    
    for batch in tqdm(batch_list(prompt_texts, batch_size), desc="Embedding Batches"):
        batch_embeddings = llm_handler.get_embeddings(batch)
        if batch_embeddings is not None and batch_embeddings.size > 0:
            all_embeddings.append(batch_embeddings)
        else:
            print(f"Warning: A batch of {len(batch)} prompts failed to embed.")

    if not all_embeddings:
        print("Failed to get any embeddings. Aborting UMAP training.")
        return
        
    embeddings = np.vstack(all_embeddings)
    print(f"Successfully generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    # 4. Fit the UMAP model
    print("Fitting UMAP model...")
    umap_model = UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        n_components=config.UMAP_N_COMPONENTS,
        random_state=42,
        verbose=True
    )
    umap_model.fit(embeddings)
    print("UMAP model fitting complete.")

    # 5. Save the trained UMAP model
    with open(config.UMAP_MODEL_PATH, 'wb') as f:
        pickle.dump(umap_model, f)
    print(f"✅ UMAP model saved to {config.UMAP_MODEL_PATH}")
    print("--- UMAP Preparation Finished ---")

if __name__ == "__main__":
    prepare_umap_model()

