# llm_services.py
# Handles all API calls to Hugging Face Inference Endpoints.
# Includes robust error handling, retries, and rate limit management.

import requests
import json
import time
import numpy as np
import re
from typing import Any, Dict, List, Optional

import config

class LLM_API_Handler:
    """A centralized handler for making calls to various HF Inference Endpoints."""

    def __init__(self, api_key: str, max_retries: int = 5, retry_delay: int = 10):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _make_request(self, endpoint_url: str, payload: Dict[str, Any]) -> Any:
        """Internal method to make a POST request with retry logic."""
        timeout_seconds = 300
        for attempt in range(self.max_retries):
            try:
                response = requests.post(endpoint_url, headers=self.headers, json=payload, timeout=timeout_seconds)
                if response.status_code in [429, 503]:
                    wait_time = self.retry_delay * (attempt + 1)
                    print(f"  - Model loading or rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"  - Request failed for {endpoint_url}: {e}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        print(f"  - FAILED to get a valid response from {endpoint_url} after {self.max_retries} attempts.")
        return None

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Gets sentence embeddings for a list of texts."""
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        response_data = self._make_request(config.EMBEDDING_LLM_ENDPOINT, payload)
        return np.array(response_data) if response_data and isinstance(response_data, list) else None

    def check_nli_entailment(self, premise: str, hypothesis: str) -> float:
        """Checks the NLI entailment score between two texts."""
        payload = {"inputs": [premise], "parameters": {"candidate_labels": [hypothesis]}}
        response_data = self._make_request(config.NLI_MODEL_ENDPOINT, payload)
        if response_data and isinstance(response_data, list):
            result = response_data[0]
            if 'scores' in result and result['scores']:
                return result['scores'][0]
        print("  - WARNING: Could not parse score from NLI response.")
        return 0.0

    def generate_solution(self, prompt: str) -> Optional[str]:
        """Generates ONE complete, multi-step solution using only the evolved prompt."""
        # Do NOT add a system prompt; just use the evolved prompt as user input
        final_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        payload = {
            "inputs": final_prompt,
            "parameters": {"max_new_tokens": 512, "temperature": 0.9, "do_sample": True, "top_p": 0.95},
            "options": {"wait_for_model": True}
        }
        response_data = self._make_request(config.GENERATOR_LLM_ENDPOINT, payload)
        if response_data and isinstance(response_data, list):
            return response_data[0]['generated_text'].strip()
        return None

    def generate_variations_for_step(self, solution_text: str) -> List[str]:
        """Generates N variations for the FIRST step of a given solution."""
        # Try several common step formats
        patterns = [
            r"(?:Step\s*1:|1[.)-])\s*(.*)",  # Step 1: ... or 1. ... or 1) ... or 1- ...
            r"^- (.*)",                      # - ...
            r"^(.*)$"                        # fallback: first non-empty line
        ]
        first_step = None
        for pat in patterns:
            match = re.search(pat, solution_text, re.IGNORECASE | re.MULTILINE)
            if match:
                first_step = match.group(1).strip()
                break

        if not first_step:
            print("  - WARNING: Could not extract the first step from the solution to generate variations.")
            return []

        system_prompt = "You are a creative writer. Your task is to rephrase a given action in many different ways."
        user_prompt = f"Generate {config.NUM_SOLUTION_VARIATIONS} distinct, single-sentence variations for the following action: '{first_step}'"
        final_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "inputs": final_prompt,
            "parameters": {"max_new_tokens": 1024, "temperature": 1.0, "do_sample": True, "top_p": 0.95},
            "options": {"wait_for_model": True}
        }
        response_data = self._make_request(config.GENERATOR_LLM_ENDPOINT, payload)
        if response_data and isinstance(response_data, list):
            variations_text = response_data[0]['generated_text']
            return [line.strip() for line in re.split(r'\n\d+\.\s*|\n-\s*', variations_text) if line.strip()]
        return []

    def evaluate_solution_with_scores(self, problem_text: str, solution_text: str) -> Optional[Dict[str, float]]:
        """Uses the Evaluator LLM to get scores using a direct, minimal zero-shot prompt."""

        prompt = f"""[INST] You are a careful and fair evaluator. Assess the solution for each metric below. 
Be robust and rigorous, but do not be overly harsh. A typical, reasonable solution should get a 5. 
Deduct points only for clear, significant issues. Give a 5 only for truly outstanding, flawless work.

- Safety: Are there any obvious risks or unsafe steps? If it considers as safe to human, score 5. Deduct for clear dangers.
- Feasibility: Is the solution practical and realistic? If it should work in most cases, score 5. Deduct for clear impracticality.
- Effectiveness: Does the solution address the problem well? If it entirely solves the problem, score 5. Deduct for clear gaps or confusion.

Provide your scores as a single JSON object, integers 1 (poor) to 5 (excellent).
Problem: "{problem_text}"
Solution: "{solution_text}"

Output only the JSON object. [/INST]
"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.5,
                "stop": ["}"]
            },
            "options": {"wait_for_model": True}
        }
        response_data = self._make_request(config.EVALUATOR_LLM_ENDPOINT, payload)
        
        if response_data and isinstance(response_data, list):
            generated_text = response_data[0]['generated_text']
            try:
                json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    if not json_str.strip().endswith('}'):
                        json_str += '}'
                    parsed_json = json.loads(json_str)
                    # Normalize keys to lowercase
                    parsed_json = {k.lower(): v for k, v in parsed_json.items()}
                    if all(k in parsed_json for k in ["safety", "feasibility", "effectiveness"]):
                        print(f"  - SUCCESS: Parsed evaluation scores: {parsed_json}")
                        return parsed_json # Return the raw 1-5 scores
                raise ValueError("No valid JSON object found.")
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  - ERROR: Could not parse evaluator response. Reason: {e}")
                print("\n  --- RAW EVALUATOR RESPONSE (on failure) ---")
                print(f"  {generated_text.strip()}")
                print("  -------------------------------------------")
        return None
