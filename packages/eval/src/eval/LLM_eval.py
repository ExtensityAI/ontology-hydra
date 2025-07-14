import json
import os
import csv
from typing import List, Optional, Dict, Any
from pydantic import Field
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class Input(LLMDataModel):
    """Input for answer matching"""
    knowledge_graph: str = Field(description="The knowledge graph with which to answer the question")
    question: str = Field(description="The question to answer")
    options: List[str] = Field(description="The possible answers to the question")

class Output(LLMDataModel):
    """Output for answer matching"""
    answer: str = Field(description="The answer to the question based solely on the provided knowledge graph")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=False,
    remedy_retry_params=dict(
        tries=25,
        delay=0.5,
        max_delay=10,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class ModelQuestionAnswer(Expression):
    def __init__(
        self,
        seed: Optional[int] = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def forward(self, input: Input) -> Output:
        if self.contract_result is None:
            return Output(answer=None)
        return self.contract_result

    def pre(self, input: Input) -> bool:
        if not isinstance(input, Input):
            raise ValueError("Input must be an Input instance!")
        return True

    def post(self, output: Output) -> bool:
        if not isinstance(output, Output):
            raise ValueError("Output must be a Output instance!")
        return True

    @property
    def prompt(self) -> str:
        return f"""[[Model Question Answer]]
Answer the question based solely on the provided knowledge graph, do not use any other information or outside knowledge.
If the question cannot be answered based solely on the provided knowledge graph, return [None].
"""


def parse_medexqa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse MedExQA JSON file and extract questions with their options.

    Args:
        file_path: Path to the MedExQA JSON file

    Returns:
        List of dictionaries containing question_id, question, options, and correct_answer
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    parsed_questions = []

    for item in data['data']:
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                # Extract question
                question = qa['question']
                question_id = qa['id']

                # Extract options from all_answers
                formatted_options = []
                raw_options = []
                correct_answer = None

                for answer in qa['all_answers']:
                    option_text = answer['text']
                    option_letter = answer['option']
                    formatted_options.append(f"{option_letter}. {option_text}")
                    raw_options.append(option_text)

                    # Find the correct answer by checking if this option matches any of the answers
                    for correct_ans in qa['answers']:
                        if correct_ans['text'].lower() in option_text.lower() or option_text.lower() in correct_ans['text'].lower():
                            correct_answer = option_text
                            break

                parsed_questions.append({
                    'question_id': question_id,
                    'question': question,
                    'formatted_options': formatted_options,
                    'raw_options': raw_options,
                    'correct_answer': correct_answer,
                    'context': paragraph['context']
                })

    return parsed_questions


def process_single_question(question_data: Dict[str, Any], kg_data: str, model: ModelQuestionAnswer) -> Dict[str, Any]:
    """
    Process a single question with the model.

    Args:
        question_data: Dictionary containing question information
        kg_data: Knowledge graph data as string
        model: ModelQuestionAnswer instance

    Returns:
        Dictionary with results including question_id, model_answer, correct_answer, and success status
    """
    try:
        input_data = Input(
            knowledge_graph=kg_data,
            question=question_data['question'],
            options=question_data['raw_options']
        )

        output = model(input=input_data)

        # Normalize the model's answer to the actual answer text
        model_answer = output.answer
        if model_answer and '. ' in model_answer:
            # If model returned "B. Synthesis", extract just "Synthesis"
            model_answer = model_answer.split('. ', 1)[1]
        elif model_answer and len(model_answer) == 1 and model_answer in 'ABCD':
            # If model returned just "B", convert to the actual answer text
            option_index = ord(model_answer) - ord('A')  # A->0, B->1, etc.
            if 0 <= option_index < len(question_data['raw_options']):
                model_answer = question_data['raw_options'][option_index]
        # If model returned full text like "Synthesis", keep it as is

        return {
            'question_id': question_data['question_id'],
            'question': question_data['question'],
            'model_answer': model_answer,
            'correct_answer': question_data['correct_answer'],
            'formatted_options': question_data['formatted_options'],
            'raw_options': question_data['raw_options'],
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'question_id': question_data['question_id'],
            'question': question_data['question'],
            'model_answer': None,
            'correct_answer': question_data['correct_answer'],
            'formatted_options': question_data['formatted_options'],
            'raw_options': question_data['raw_options'],
            'success': False,
            'error': str(e)
        }


def test(max_workers: int = 4):
    """Test the ModelQuestionAnswer contract with MedExQA data using parallel processing."""
    print(f"Starting evaluation with {max_workers} workers...")

    # Parse MedExQA data
    medexqa_path = "/Users/ryang/Work/ExtensityAI/research-ontology/MedExQA/test/biomedical_engineer_test.json"
    questions = parse_medexqa_data(medexqa_path)

    print(f"Parsed {len(questions)} questions from MedExQA dataset")

    # Load knowledge graph from file
    kg_path = "/Users/ryang/Work/ExtensityAI/research-ontology/eval/runs/run_gpt-4.1-mini/biomedical_engineer/topics/Biomedical Engineer/kg.json"
    with open(kg_path, 'r') as f:
        kg_data = json.load(f)

    kg_data_str = str(kg_data)

    # Create model instance
    model = ModelQuestionAnswer(seed=42)

    # Process questions in parallel
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(process_single_question, question, kg_data_str, model): question
            for question in questions
        }

        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_question):
            result = future.result()
            results.append(result)
            completed += 1

            # Print progress
            print(f"Completed {completed}/{len(questions)} questions ({(completed/len(questions)*100):.1f}%)")

            if result['success']:
                print(f"  Q{result['question_id']}: Model answered '{result['model_answer']}', Correct: '{result['correct_answer']}'")
            else:
                print(f"  Q{result['question_id']}: Error - {result['error']}")

    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Total questions: {len(questions)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per question: {total_time/len(questions):.2f} seconds")

    # Calculate accuracy for successful runs
    successful_results = [r for r in results if r['success']]
    if successful_results:
        correct_answers = sum(1 for r in successful_results if r['model_answer'] == r['correct_answer'])
        accuracy = correct_answers / len(successful_results) * 100
        print(f"Accuracy: {accuracy:.1f}% ({correct_answers}/{len(successful_results)})")

        # Save results to CSV file
    model_eval_path = os.path.join(os.path.dirname(kg_path), "model_eval")
    os.makedirs(model_eval_path, exist_ok=True)
    output_file = os.path.join(model_eval_path, "model_evaluation_results.csv")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'question_id',
            'question',
            'model_answer',
            'correct_answer',
            'is_correct',
            'success',
            'error',
            'options'
        ])

                # Write data rows
        for result in results:
            is_correct = result['model_answer'] == result['correct_answer'] if result['success'] else False
            options_str = ' | '.join(result['formatted_options'])

            writer.writerow([
                result['question_id'],
                result['question'],
                result['model_answer'] or '',
                result['correct_answer'] or '',
                is_correct,
                result['success'],
                result['error'] or '',
                options_str
            ])

    print(f"Results saved to: {output_file}")

    # Also save summary as JSON for metadata
    summary_file = os.path.join(model_eval_path, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': {
                'total_questions': len(questions),
                'successful': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'total_time': total_time,
                'accuracy': accuracy if successful_results else 0
            }
        }, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    test(max_workers=5)

