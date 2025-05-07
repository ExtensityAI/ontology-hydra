import openai
from tqdm import tqdm

from ontopipe.cqs.utils import MODEL
from ontopipe.models import KG, KGState, Ontology

_system_prompt = """You are an expert in knowledge graph construction and semantic triplet extraction. Your task is to extract semantic triplets from text and format them as a consistent, ontology-compliant knowledge graph.

### Output Format
Return a JSON array of triplets, each containing:
- "subject": Entity performing the action (PascalCase)
- "predicate": Relationship or action (camelCase) 
- "object": Entity receiving the action (PascalCase)

### Entity and Relationship Rules
1. Entity Consistency: If an entity exists in the current knowledge graph, ALWAYS use the exact same name
2. Entity Naming: For new entities, choose concise, descriptive PascalCase names
3. Predicate Naming: All predicates must be in camelCase and MUST exist in the provided ontology
4. Entity Typing: EVERY entity requires a type declaration using the "isA" predicate
5. Type Validity: Entity types MUST be defined in the ontology

### Quality Standards
1. Extract only factual relationships explicitly stated or directly inferable from the text
2. Resolve coreferences (pronouns, repeated mentions) to maintain consistent entity references
3. Prefer specificity over generality when appropriate
4. For ambiguous cases, choose the interpretation that maintains graph consistency
5. Never invent relationships not supported by the text or ontology

### Triple Extraction Process
1. First identify all entities in the text
2. Determine the appropriate type for each entity based on the ontology
3. Create "isA" triplets for all entities
4. Extract all relationships between entities that conform to the ontology
5. Verify all triplets are consistent with the existing knowledge graph

<ontology>{ontology}</ontology>

<kg>{kg}</kg>

Respond ONLY with the JSON array of triplets. Do not include any explanations or additional text in your response."""

# TODO make this multi-round to allow for extraction of triplets whose type might not be known at the start?


def generate_kg(
    texts: list[str],
    kg_name: str,
    ontology: Ontology,
    batch_size: int = 1,
    max_retries: int = 5,
) -> KG:
    state = KG(name=kg_name, triplets=[])
    entity_types = {}  # Track entities and their types

    for i in tqdm(range(0, len(texts), batch_size)):
        text = "\n".join(texts[i : i + batch_size])

        retry_count = 0
        success = False
        errors = []

        while not success and retry_count <= max_retries:
            print("Current try:", retry_count + 1)
            print("Errors so far:", errors)
            try:
                # Prepare the base message
                messages = [
                    {
                        "role": "system",
                        "content": _system_prompt.format(
                            kg=state.model_dump_json(), ontology=ontology.model_dump_json()
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"<text>{'\n'.join(text)}</text>",
                    },
                ]

                # If this is a retry, add error feedback
                if retry_count > 0:
                    error_message = f"""
Your previous response had the following errors:
{", ".join(errors)}

Please fix these errors and provide a corrected set of triplets that follow all rules in the ontology.
"""
                    messages.append({"role": "user", "content": error_message})

                response = openai.beta.chat.completions.parse(
                    model=MODEL,
                    messages=messages,
                    response_format=KGState,
                )

                new_triplets = response.choices[0].message.parsed
                if new_triplets is None:
                    raise ValueError("Failed to generate triplets")

                errors = []  # Reset errors for this pass

                # First pass: collect all isA relationships
                for triplet in new_triplets.triplets:
                    if triplet.predicate == "isA":
                        # Check 1: isA objects must be valid classes
                        if not ontology.has_class(triplet.object):
                            errors.append(f"Type '{triplet.object}' is not defined in the ontology")
                        else:
                            entity_types[triplet.subject] = triplet.object

                # Second pass: validate all triplets
                for triplet in new_triplets.triplets:
                    # Check 2: Subjects cannot be ontology classes
                    if ontology.has_class(triplet.subject):
                        errors.append(
                            f"Subject '{triplet.subject}' is an ontology class and cannot be used as a subject"
                        )
                        continue

                    # For non-isA triplets, validate the predicate and object
                    if triplet.predicate != "isA":
                        # Check that predicate exists in ontology
                        if not ontology.has_property(triplet.predicate):
                            errors.append(f"Predicate '{triplet.predicate}' is not defined in the ontology")
                            continue

                        # Objects that aren't ontology classes need type definitions
                        if not ontology.has_class(triplet.object) and triplet.object not in entity_types:
                            errors.append(f"Entity '{triplet.object}' has no type definition")
                            continue

                # Check 3: All subjects must have type definitions
                entities = {triplet.subject for triplet in new_triplets.triplets}
                entities.update(
                    triplet.object
                    for triplet in new_triplets.triplets
                    if triplet.predicate != "isA" and not ontology.has_class(triplet.object)
                )

                for entity in entities:
                    if entity not in entity_types and not ontology.has_class(entity):
                        errors.append(f"Entity '{entity}' has no type definition")

                # If we have errors, raise an exception to trigger retry
                if errors:
                    raise ValueError("Validation errors found")

                # If we got here, there were no errors - add triplets to state
                for triplet in new_triplets.triplets:
                    # Skip duplicate isA statements
                    if triplet.predicate == "isA" and any(
                        t.predicate == "isA" and t.subject == triplet.subject for t in state.triplets
                    ):
                        continue

                    state.triplets.append(triplet)

                success = True

            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Failed after {max_retries} retries. Last error: {str(e)}")
                    # Optionally log the text that failed
                    print(f"Problematic text: {text[:100]}...")

        if not success:
            print("Warning: Unable to process some text after maximum retries")

    return state
