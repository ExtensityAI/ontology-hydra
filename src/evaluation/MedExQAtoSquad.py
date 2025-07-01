import json
import csv

TITLE = "Biomedical Engineering"
VERSION = "v1.0"

def parse_medexqa_to_json(input_file, output_file):
    # Create the outer structure matching SQuAD format
    squad_data = {
        "version": VERSION,
        "data": [
            {
                "title": TITLE,
                "paragraphs": []
            }
        ]
    }

    paragraphs = []

    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip empty lines and read TSV
        tsv_reader = csv.reader((line for line in f if line.strip()), delimiter='\t')

        for row in tsv_reader:
            if len(row) < 7:  # Skip malformed rows
                continue

            question = row[0]
            options = [row[1], row[2], row[3], row[4]]
            explanation = row[5]
            detailed_explanation = row[6]
            correct_answer = row[7]  # A, B, C, or D

            # Convert letter answer to index (0-3)
            answer_idx = ord(correct_answer) - ord('A')
            correct_answer_text = options[answer_idx]

            # Combine both explanations for context
            context = f"{explanation} {detailed_explanation}"

            # Set answer_start to 0 if answer not found in context
            try:
                # Find all occurrences of the answer in the context
                answer_start = context.index(correct_answer_text)
            except ValueError:
                # If exact match not found, try case-insensitive search
                lower_context = context.lower()
                lower_answer = correct_answer_text.lower()
                try:
                    answer_start = lower_context.index(lower_answer)
                    # Use the original case from the context
                    correct_answer_text = context[answer_start:answer_start + len(correct_answer_text)]
                except ValueError:
                    # If still not found, default to -1
                    answer_start = -1

            # Create 4 identical answers as seen in SQuAD format
            answers = [
                {"text": correct_answer_text, "answer_start": answer_start}
                for _ in range(4)
            ]

            qa_item = {
                "question": question,
                "id": f"medexqa_{len(paragraphs)}",
                "answers": answers,
                "is_impossible": False,
                "all_answers": [
                    {"text": option, "option": chr(ord('A') + i)}
                    for i, option in enumerate(options)
                ]
            }

            # Add to paragraphs
            paragraph = {
                "context": context,
                "qas": [qa_item]
            }
            paragraphs.append(paragraph)

    # Add paragraphs to the data structure
    squad_data["data"][0]["paragraphs"] = paragraphs

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "MedExQA/test/biomedical_engineer_test.tsv"
    output_file = "MedExQA/test/biomedical_engineer_test.json"
    parse_medexqa_to_json(input_file, output_file)
