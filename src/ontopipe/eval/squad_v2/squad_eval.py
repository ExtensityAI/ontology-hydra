from evaluate import load
import json

DATA_DIR = "data/squad_2.0_Normans"

# Load predictions and references from JSON file
with open(f"{DATA_DIR}/qa_predictions_baseline_context_parts.json") as f:
    data = json.load(f)

# Format predictions as expected by the squad_v2 metric
predictions = []
for item in data['predictions']:
    predictions.append({
        'id': item['id'],
        'prediction_text': item.get('prediction_text', ''),
        'no_answer_probability': 0.5
    })

# Format references as expected by the squad_v2 metric
references = []
for item in data['references']:
    answers = [{'text': ans['text'], 'answer_start': ans['answer_start']} for ans in item['answers']]
    references.append({
        'id': item['id'],
        'answers': answers
    })

# Compute metrics
squad_metric = load("squad_v2")
results = squad_metric.compute(predictions=predictions, references=references)

print("SQuAD v2 Metric Results:")
print(results)