import json

def contains(text):
    return "mockingbird" in text.lower()

def extract(data):
    extracted = []

    for entry in data:
        if contains(entry.get("title", "")):
            # Include the entire entry if title matches
            extracted.append(entry)

    return extracted

def main():
    input_file = "src/evaluation/train-v2.0.json"
    output_file = "mockingbird_extracted.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = {
        "version": data.get("version", "v2.0"),
        "data": extract(data["data"])
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Extracted entries saved to '{output_file}'.")

if __name__ == "__main__":
    main()