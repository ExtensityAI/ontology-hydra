import openai

from ontopipe.cqs.comittee import ComitteeMember
from ontopipe.cqs.utils import MODEL

generate_questions_prompt = """
You represent a group of different experts as stated below in <group/> tags.. You have been presented with a scope document for an ontology in the domain of <domain>{domain}</domain>.

## Your Task
Based on your expertise and the provided scope document, generate a list of questions that you  would want answered by a comprehensive ontology in this domain.

## Guidelines for Question Generation
1. Focus on questions that are important to you as an expert in this field
2. Consider questions about:
   * Key concepts and their definitions
   * Important distinctions between related terms
   * Essential classifications or categorizations
   * Critical attributes or properties
   * Domain-specific constraints or rules

## Output Format
1. Organize your questions into a simple list format (e.g. - Question 1)
2. Use bullet points for individual questions
3. Ensure each question is specific and concrete
4. Phrase questions to elicit detailed, precise answers

Generate only questions that you, as this specific expert persona, would consider relevant and important for understanding the domain. Do not include questions outside your area of expertise.

<group>{group}</group>
<scope_document>{scope_document}</scope_document>
"""


def generate_questions(domain: str, group: list[ComitteeMember], scope_document: str):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": generate_questions_prompt.format(
                    domain=domain,
                    group="\n".join([f"- {p.persona.description}" for p in group]),
                    scope_document=scope_document,
                ),
            }
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to generate questions")

    # split response into lines, filter out empty lines and lines that don't start with "-", and strip the leading "-"
    questions = response.choices[0].message.content.strip().split("\n")
    return [q[1:].strip() for q in questions if q.strip().startswith("-")]
