import json

DATA_DIR = "src/ontopipe/eval/data/squad_2.0_Normans"

# Load predictions and references from JSON file
with open(f"{DATA_DIR}/qa_predictions_baseline_context_parts.json") as f:
    data = json.load(f)

# Format predictions as expected by the squad_v2 metric


print("SQuAD v2 Metric Results:")
print(results)
