# test_evaluator.py
# A comprehensive test suite for the evaluation pipeline.
# Includes a live-fire test for the Evaluator LLM and a local unit test for the Semantic Entropy logic.

import unittest
from unittest.mock import MagicMock
import math
import time

import config
from llm_services import LLM_API_Handler
from evaluation import SemanticEntropyCalculator, LexicalDiversityCalculator

# --- Unit Test for Local Logic (No API Calls) ---

class TestSemanticEntropyLogic(unittest.TestCase):
    """
    Tests the internal logic of the SemanticEntropyCalculator using mock data.
    This does NOT make real API calls.
    """
    def setUp(self):
        self.mock_llm_handler = MagicMock(spec=LLM_API_Handler)

    def test_entropy_high_diversity(self):
        """Tests if semantically different sentences produce a high entropy score."""
        print("\n--- Unit Test: Semantic Entropy (High Diversity) ---")
        variations = ["Tie the rope.", "Scoop the water.", "Shine the light."]
        
        def mock_nli(premise, hypothesis): return 0.1 if premise != hypothesis else 1.0
        self.mock_llm_handler.check_nli_entailment.side_effect = mock_nli

        calculator = SemanticEntropyCalculator(self.mock_llm_handler)
        entropy = calculator.calculate_entropy(variations)
        
        expected = -sum((1/3) * math.log2(1/3) for _ in range(3))
        self.assertAlmostEqual(entropy, expected, places=4)
        print(f"✅ PASSED: High diversity entropy is {entropy:.4f} (correct)")

    def test_entropy_low_diversity(self):
        """Tests if semantically similar sentences produce a low entropy score."""
        print("\n--- Unit Test: Semantic Entropy (Low Diversity) ---")
        variations = ["Attach the string.", "Tie the line.", "Get the bucket."]
        
        def mock_nli(premise, hypothesis):
            similar = {"Attach the string.", "Tie the line."}
            return 0.95 if premise in similar and hypothesis in similar else 0.1
        self.mock_llm_handler.check_nli_entailment.side_effect = mock_nli

        calculator = SemanticEntropyCalculator(self.mock_llm_handler)
        entropy = calculator.calculate_entropy(variations)

        p1, p2 = 2/3, 1/3
        expected = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        self.assertAlmostEqual(entropy, expected, places=4)
        print(f"✅ PASSED: Low diversity entropy is {entropy:.4f} (correct)")

# --- Live Test for Evaluator LLM (Makes Real API Calls) ---

def run_live_evaluator_test():
    """
    Sends several hand-crafted solutions to the Evaluator LLM and prints the results.
    """
    print("\n\n--- Starting Live Evaluator LLM Test ---")

        
    llm_handler = LLM_API_Handler(config.HF_API_KEY)

    problem = "You have dropped your keys down a deep, narrow drain. You have a magnet, a piece of string, and a plastic cup."

    # --- NEW: Expanded test cases with more variety and extreme examples ---
    test_cases = [
        {
            "name": "✅ Perfect, Logical Solution",
            "solution": "Step 1: Tie the string securely to the magnet. Step 2: Lower the magnet into the drain to attract the keys. Step 3: Carefully pull the keys up."
        },
        {
            "name": "💥 Extremely Unsafe Solution",
            "solution": "Step 1: Pour corrosive acid down the drain to dissolve the grate. Step 2: Reach in and grab the keys."
        },
        {
            "name": "🤔 Safe but Completely Ineffective Solution",
            "solution": "Step 1: Shout down the drain to ask the keys to come back up."
        },
        {
            "name": "🤪 Nonsensical and Physically Impossible Solution",
            "solution": "Step 1: Eat the magnet to gain magnetic powers. Step 2: Use your new powers to telekinetically lift the keys."
        },
        {
            "name": "⚠️ Clever but Flawed Solution",
            "solution": "Step 1: Melt the plastic cup into a long, sticky rod. Step 2: Use the rod to fish out the keys."
        },
        {
            "name": "🧑‍🔬 Overly Complicated Rube Goldberg Solution",
            "solution": (
                "Step 1: Use the string to create a pulley system with the plastic cup as a counterweight. "
                "Step 2: Attach the magnet to the end of the string. "
                "Step 3: Carefully maneuver the magnet into the drain using the pulley system to retrieve the keys."
            )
        },
        {
            "name": "🧒 Childlike Imagination Solution",
            "solution": (
                "Step 1: Pretend the magnet is a superhero and ask it to fly down and rescue the keys. "
                "Step 2: Wait for the keys to magically appear in your hand."
            )
        },
        {
            "name": "🔬 Scientific but Impractical Solution",
            "solution": (
                "Step 1: Fill the drain with liquid nitrogen to freeze the keys. "
                "Step 2: Use the magnet to attract the frozen keys and pull them up."
            )
        },
        {
            "name": "🛠️ Realistic but Tedious Solution",
            "solution": (
                "Step 1: Lower the plastic cup tied to the string to scoop out any water. "
                "Step 2: Repeat until the drain is dry. "
                "Step 3: Use the magnet tied to the string to retrieve the keys."
            )
        },
        {
            "name": "🧙 Magical Thinking Solution",
            "solution": (
                "Step 1: Cast a spell on the string to make it snake down the drain and tie itself to the keys. "
                "Step 2: Pull the string to retrieve the keys."
            )
        },
        {
            "name": "🧑‍🚒 Call for Help Solution",
            "solution": (
                "Step 1: Use the string to tie a note to the magnet. "
                "Step 2: Lower the note into the drain asking for help. "
                "Step 3: Wait for someone to find the note and assist you."
            )
        },
        {
            "name": "🧹 Use Available Tools Creatively",
            "solution": (
                "Step 1: Flatten the plastic cup and attach it to the magnet with the string. "
                "Step 2: Lower the contraption into the drain to both scoop and attract the keys. "
                "Step 3: Pull up once you feel resistance."
            )
        },
        {
            "name": "🧊 Wait for Environmental Change",
            "solution": (
                "Step 1: Wait for it to rain heavily so the water rises and floats the keys up. "
                "Step 2: Grab the keys when they are within reach."
            )
        }
    ]

    print(f"\nTesting with problem: '{problem}'")
    print("="*50)

    #all_tests_passed = True
    received_scores = []

    for i, case in enumerate(test_cases):
        print(f"\n--- Testing Case: {case['name']} ---")
        print(f"Solution: {case['solution']}")
        
        time.sleep(2) 
        
        scores = llm_handler.evaluate_solution_with_scores(problem, case['solution'])
        
        if scores and all(k in scores for k in ["safety", "feasibility", "effectiveness"]):
            received_scores.append(scores)
        else:
            print("❌ FAILED! Did not receive valid scores for this case.")
            #all_tests_passed = False
        print("-"*50)

    # --- Final Verification ---
    if len(received_scores) == len(test_cases):
        first_score_set = set(f"{s['safety']},{s['feasibility']},{s['effectiveness']}" for s in received_scores)
        
        if len(first_score_set) == 1:
            print("\n🔥 TEST FAILED: The evaluator returned the exact same scores for every test case.")
            #all_tests_passed = False
        else:
            print("\n🎉 TEST PASSED: The evaluator produced varied scores for different solutions.")
    else:
        print("\n🔥 TEST FAILED: Not all test cases produced valid scores.")
        #all_tests_passed = False

# --- Divergent Creativity (Semantic Entropy) Test Cases ---

divergent_entropy_cases = [
    {
        "name": "🔁 Low Divergence (All Variations Nearly Identical)",
        "variations": [
            "Tie the string to the magnet and lower it to get the keys.",
            "Tie the string to the magnet and lower it to get the keys.",
            "Tie the string to the magnet and lower it to get the keys."
        ]
    },
    {
        "name": "🔗 Low Divergence (Paraphrased Variations)",
        "variations": [
            "Attach the string to the magnet and retrieve the keys.",
            "Secure the magnet to the string and use it to pull up the keys.",
            "Fasten the string to the magnet, lower it, and collect the keys."
        ]
    },
    {
        "name": "🌈 High Divergence (Distinct Approaches)",
        "variations": [
            "Use the magnet tied to the string to fish out the keys.",
            "Melt the plastic cup into a hook and try to scoop the keys.",
            "Wait for rain to flood the drain and float the keys up."
        ]
    },
    {
        "name": "🌀 Maximal Divergence (Wildly Different, Imaginative)",
        "variations": [
            "Summon a magical wind to blow the keys out of the drain.",
            "Ask a passing bird to fly down and pick up the keys.",
            "Invent a shrinking device, shrink yourself, and retrieve the keys."
        ]
    },
    {
        "name": "🧩 Mixed Divergence (Two Similar, One Different)",
        "variations": [
            "Tie the string to the magnet and lower it to get the keys.",
            "Attach the string to the magnet and retrieve the keys.",
            "Pour water into the drain until the keys float to the top."
        ]
    },
    {
        "name": "📚 Long, Paraphrased Solutions (Moderate Diversity)",
        "variations": [
            "Carefully tie the string to the magnet, ensuring a secure knot, then lower the magnet into the drain and patiently maneuver it until the keys are attracted and can be pulled up.",
            "Fasten the string tightly around the magnet, gently lower the contraption into the drain, and slowly move it around until the keys stick to the magnet, then retrieve them.",
            "Secure the magnet to one end of the string, lower it into the drain, and carefully guide it until the keys are caught, then pull the string up to recover the keys."
        ]
    },
    {
        "name": "🧠 Highly Diverse, Long Solutions (Maximal Diversity)",
        "variations": [
            "Construct a makeshift fishing rod by attaching the magnet to the string, then use the plastic cup as a handle to lower the magnet into the drain, skillfully maneuvering it until the keys are retrieved.",
            "Melt the plastic cup to form a flexible hook, attach it to the string, and attempt to snag the keys by carefully lowering the hook into the drain and pulling up once you feel resistance.",
            "Wait for a heavy rainstorm to flood the drain, causing the keys to float upwards, then use the magnet tied to the string to attract and retrieve the keys from the surface."
        ]
    },
    {
        "name": "🌌 Wildly Imaginative, Long Solutions",
        "variations": [
            "Summon a magical wind by chanting an ancient incantation, directing the breeze to gently lift the keys out of the drain and into your waiting hand, then thank the wind for its assistance.",
            "Ask a passing bird to fly down into the drain, grasp the keys in its beak, and deliver them to you, rewarding the bird with a song and a handful of seeds.",
            "Invent a shrinking device using the plastic cup and magnet, reduce yourself to the size of a mouse, climb into the drain, pick up the keys, and return to normal size outside."
        ]
    },
    {
        "name": "🔀 Mixed Approaches, Long Solutions",
        "variations": [
            "Tie the string to the magnet, lower it into the drain, and carefully move it around until the keys are attached, then slowly pull them up, making sure not to drop them back in.",
            "Pour water into the drain using the plastic cup until the keys float to the top, then use the magnet tied to the string to attract and retrieve them.",
            "Create a pulley system with the string and plastic cup, attach the magnet, and use the system to lower and maneuver the magnet until the keys are caught and can be lifted out."
        ]
    }
]

def run_divergent_entropy_test():
    print("\n--- Divergent Creativity (Semantic Entropy & Lexical Diversity) Test ---")
    llm_handler = LLM_API_Handler(config.HF_API_KEY)
    calculator = SemanticEntropyCalculator(llm_handler)
    lex_div_calc = LexicalDiversityCalculator(segment_length=5)  # Add this line

    for case in divergent_entropy_cases:
        print(f"\nCase: {case['name']}")
        entropy = calculator.calculate_entropy(case['variations'])
        msttr = lex_div_calc.msttr(case['variations'])  # Add this line
        print(f"Entropy: {entropy:.4f}")
        print(f"MSTTR (Lexical Diversity): {msttr:.4f}")  # Add this line
        for v in case['variations']:
            print(f"  - {v}")

# Call this function in your __main__ block if desired


if __name__ == "__main__":
    # Run the local unit tests first
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSemanticEntropyLogic))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    # If local tests pass, run the live API test
    if result.wasSuccessful():
        run_live_evaluator_test()
    else:
        print("\nSkipping live evaluator test because local unit tests failed.")

    # Run the divergent entropy test
    run_divergent_entropy_test()
