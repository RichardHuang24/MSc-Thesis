import pandas as pd
import json

csv_path = "/Users/richardhuang/Desktop/thesis/DATASET/MacGyver/problem_solution_pair.csv"
output_path = "/Users/richardhuang/Desktop/thesis/data/macgyver_unconventional_50.json"

df = pd.read_csv(csv_path)

# Filter for Yes Solvable and Unconventional
filtered = df[(df['Solvable?'] == 'Yes') & (df['Unconventional?'] == 'unconventional')]

# Take only the first 50
filtered = filtered.head(50)

# Build the list of dicts in the desired format
data = []
for idx, row in filtered.iterrows():
    entry = {
        "id": f"macgyver_{row['ID']}",
        "problem_text": row["Problem"],
        "ground_truth_solution": row["Solution"]
    }
    data.append(entry)

# Save to JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} entries to {output_path}")