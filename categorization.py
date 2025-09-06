import json
import re

# filepath: /Users/richardhuang/Desktop/thesis/data/macgyver_dataset.json
with open("data/macgyver_dataset.json") as f:
    problems = json.load(f)

def categorize(problem_text):
    text = problem_text.lower()
    # Indoors/Household
    if re.search(r"\b(bedroom|closet|storage|cooking|dish|dining room|fitness|gym|sports|hair styling|home improvement|hotel|indoors arrangement|kitchen|library|living room|office|work|packing|personal grooming|beauty routine|shopping)\b", text):
        return "Indoors/Household"
    # Neutral
    if re.search(r"\b(party|classroom|university|lecture hall|dog training|garage|going out for a meal|plants|flowers|garden|public speaking|recycling|waste management|school|student activity|science fair|science laboratory|swimming|university campus|vehicle maintenance|weather preparation|response)\b", text):
        return "Neutral"
    # Outdoors
    if re.search(r"\b(beach|vine|beach|boat|campsite|city streets|sidewalks|construction work|desert survival|exploring a cave|farm duties|forest|jungle|hiking|camping|traveling|parks|rain|winter|zoo|playground|playing with snow|playing with water|rooftop terrace)\b", text):
        return "Outdoors"
    return "Neutral"  # Default to Neutral if not matched

output = []
for prob in problems:
    label = categorize(prob.get("problem_text", ""))
    output.append({
        "problem_id": prob.get("id", ""),
        "problem_text": prob["problem_text"],
        "category": label
    })

with open("macgyver_problem_categories.json", "w") as f:
    json.dump(output, f, indent=2)