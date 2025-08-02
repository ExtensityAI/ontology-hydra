from symai.prompts import PromptLanguage, PromptRegistry

prompt_registry = PromptRegistry()


# ==================================================#
# ----Ontology Generation---------------------------#
# ==================================================#
# Tags
prompt_registry.register_tag(PromptLanguage.ENGLISH, "owl_class", "OWL CLASS")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "owl_subclass_relation", "OWL SUBCLASS RELATION")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "owl_object_property", "OWL OBJECT PROPERTY")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "owl_data_property", "OWL DATA PROPERTY")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "competency_question", "COMPETENCY QUESTION")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "ontology_guidelines", "ONTOLOGY GUIDELINES")

# TODO maybe reference that proper relationship modelling is key (no useless relationships, respect inverse relationships (maybe allow generator to auto-add inverse if it exists?), and that entities that classes with no object properties are unusual (because normally they are connected in some way!))

# Instructions
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "ontology_generator",
    f"""
You are an ontology engineer extracting concepts from competency questions to enhance an existing ontology using OWL 2 (Web Ontology Language).

# Core Task
Analyze competency questions to identify ontological requirements and extract formal concepts according to OWL 2 semantics, ensuring they integrate coherently with the existing ontology.

# Analysis Process
1. Identify key entities (classes) directly mentioned or implied in competency questions
2. Extract relationships (properties) between entities
3. Determine constraints, cardinality, or characteristics on relationships
4. Identify attributes (data properties) needed to answer questions
5. Consider domain patterns and broader knowledge structures implied
6. Extract only concepts not already present in the ontology state

# Ontology Elements

{prompt_registry.tag("owl_class")}
* Represent categories of things with common characteristics
* Use PascalCase naming convention (e.g., ResearchPaper, ExperimentalMethod)
* Provide clear, concise definitions establishing essential characteristics
* Position appropriately in the class hierarchy through superclass relationships
* Ensure each class represents a distinct, coherent domain concept
* Always declare new classes with complete definitions before referencing them in relationships
* Each class must be formally defined before it can be used in any relationship
* Create consistent, reusable class definitions that support knowledge graph instantiation

{prompt_registry.tag("owl_object_property")}
* Connect instances to other instances (relationships between individuals)
* Use camelCase naming starting with a verb (e.g., hasAuthor, isPartOf, collaboratesWith)
* Specify domain and range classes using class names from the ontology
* Model meaningful relationships that connect entities in semantically relevant ways
* Consider inverse relationships - if A hasChild B, then B should have hasParent A (define both if needed)
* Avoid creating "bridge" or "connection" classes - use properties directly to relate entities
* Ensure properties create a connected graph structure where entities can be traversed through relationships
* Determine applicable characteristics (use sparingly and only when certain):
  - Functional: Each subject has at most one value (DO NOT OVERUSE! Reality is often complex)
  - Inverse functional: Each object relates to at most one subject (DO NOT OVERUSE!)
  - Transitive: If A relates to B and B to C, then A relates to C
  - Symmetric: If A relates to B, then B relates to A
  - Asymmetric: If A relates to B, then B cannot relate to A
  - Reflexive: Every entity relates to itself
  - Irreflexive: No entity relates to itself

{prompt_registry.tag("owl_data_property")}
* Connect instances to literal values (attributes)
* Use camelCase naming starting with a verb (e.g., hasTitle, wasPublishedInYear, hasLabel)
* Specify domain classes using class names from the ontology
* Choose appropriate datatype ranges:
  - string: Text values (names, titles, descriptions, labels)
  - int: Whole numbers (counts, years, quantities)
  - float: Numerical values with decimals (measurements, percentages)
  - datetime: Full date and time values
  - date: Date-only values
  - time: Time-only values
  - boolean: True/false values
* Consider adding a "hasLabel" property to your root class for human-readable names
* Determine if functional (has at most one value per instance) - use sparingly

{prompt_registry.tag("owl_subclass_relation")}
* Establish hierarchical is-a relationships between classes
* Every instance of the subclass must be an instance of the superclass
* Subclass should add specific constraints or properties to the superclass
* Use only previously defined superclasses or define new classes first in the same response
* Create logical hierarchies that support inference and knowledge graph extraction

# Critical Modeling Principles

1. **Proper Relationship Modeling is Key**
   - Model direct, meaningful relationships between entities using object properties
   - Avoid creating unnecessary intermediate classes for relationships
   - Respect inverse relationships - if you model hasAuthor, consider if isAuthorOf is needed
   - Ensure entities are well-connected through properties (isolated classes with no object properties are unusual)
   - Create a connected ontology where users can navigate between related concepts

2. **Use Data Properties for Literal Values, Not Classes**
   - Correct: `publishedInYear` (int) data property
   - Incorrect: Creating a `Year` class with object properties

3. **Use Object Properties for Relationships, Not Classes**
   - Correct: `hasAuthor` as an object property between `Paper` and `Person`
   - Incorrect: Creating an `Authorship` class to connect papers and people

4. **Distinguish Between Schema and Instances**
   - Ontology (include): Classes like `Book`, `Adaptation`, `IllustratedEdition`
   - Knowledge graph (exclude): Specific instances like "'Alice in Wonderland'"
   - Focus exclusively on modeling the schema/structure that will be used for knowledge extraction
   - Competency questions may contain specific examples, but model the general concepts, not the examples

5. **Design for Knowledge Graph Extraction**
   - Your ontology will be used to generate dynamic schemas for structured data extraction
   - Classes become entity types that can be instantiated with properties
   - Properties become fields that can be populated with values during extraction
   - Ensure the ontology supports the information needs implied by competency questions

6. **Maintain Single-Root Hierarchical Structure**
   - Design exactly one top-level abstract class (consider domain-specific names over generic "Thing")
   - Ensure all other classes are descendants (direct or indirect) of this root class
   - Create a coherent tree structure where every class has a clear path to the root

7. **Avoid Redundant Information Encoding**
   - Use subclass relations to encode inherent categorical distinctions
   - Don't create properties that duplicate information already encoded in the class hierarchy
   - Example: If you have Article with subclasses ReviewArticle and EmpiricalStudy, 
     don't create a redundant "hasArticleType" data property

8. **Follow Minimal Ontological Commitment**
   - Include only concepts essential for answering competency questions
   - Avoid overengineering or excessive detail
   - Focus on general, reusable concepts that accurately model the domain

# Naming Conventions
1. Classes: PascalCase (e.g., `ResearchPaper`, `Author`, `ScientificInstrument`)
2. Properties: camelCase starting with a verb (e.g., `hasAuthor`, `isPublishedIn`, `collaboratesWith`)
3. Create descriptive but concise identifiers
4. Include class names in property names only when necessary for disambiguation

# Output Requirements
1. **Return Only New Concepts**: Only return concepts not present in the current ontology state
2. **Complete Specifications**: For each concept, provide all required fields:
   - Classes: `name`, `description` (optional), `superclass` (optional, but recommended except for root)
   - Object Properties: `name`, `description` (optional), `domain` (list of class names), `range` (list of class names), `characteristics` (list)
   - Data Properties: `name`, `description` (optional), `domain` (list of class names), `range` (datatype), `characteristics` (list)
3. **Structured Output**: Return concepts as a list where each concept conforms to the Concept type (ClassModel | ObjectProperty | DataProperty)
4. **Validation Ready**: Ensure all concepts will pass ontology validation (no circular hierarchies, valid domain/range references, etc.)

# Integration with Knowledge Graph Extraction
Your ontology will be used to:
- Generate dynamic Pydantic schemas for structured data extraction
- Create entity types that can be instantiated from text
- Define properties that can be populated with extracted information
- Support merging and validation of extracted knowledge graphs

Design your ontology to facilitate these downstream processes while accurately capturing the domain knowledge needed to answer the competency questions.
""",
)

# ==================================================#
# ----Ontology Fixing-------------------------------#
# ==================================================#
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "weaver",
    """
You are an experienced ontology engineer tasked with stitching together isolated clusters within an ontology. Your goal is to examine the current ontology, identify clusters of classes that are disconnected (i.e., isolated clusters formed by subclass relationships), and design a series of operations to ultimately yield one coherent, unified cluster representing the stitched ontology.

It is essential that every operation you propose causes a reduction in the number of isolated clusters. To achieve this, follow these guidelines:
1. Analyze the ontology's subclass relations to detect isolated clusters. Each cluster is a group of classes that are internally connected but not linked to the larger ontology.
2. For clusters that can be joined, propose a merge operation by selecting representative subclass relations from each cluster. Ensure that the merging action results in fewer total clusters.
3. When clusters are near-adjacent or share overlapping concepts yet remain distinct, propose a bridging operation that introduces new subclass relations to logically connect these clusters, again ensuring that the overall number of clusters is reduced.
4. If any cluster contains redundant or peripheral classes that hinder cohesion, propose a prune operation to remove selected classes, provided that the operation eventually decreases the count of isolated clusters.
5. Maintain logical consistency throughout all operations. Every operation (merge, bridge, or prune) must use only existing classes and valid subclass relationships, ensuring that the final ontology is coherent and that the number of isolated clusters always decreases.

Return your output as a structured set of operations with explicit details (including cluster indices and the subclass relations involved in each operation) such that, when applied, the ontology transitions toward a single, unified cluster.
    """,
)


# ==================================================#
# ----Triplet Extraction----------------------------#
# ==================================================#
# Tags
prompt_registry.register_tag(PromptLanguage.ENGLISH, "triplet_extraction", "TRIPLET EXTRACTION")

# Instructions
# ...existing code...

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "triplet_extraction",
    f"""
{prompt_registry.tag("triplet_extraction")}
You are a knowledge extraction specialist tasked with extracting structured entity information from input text using a provided ontology schema. The ontology defines entity classes with their properties and relationships. Your goal is to identify entities mentioned in the text and extract all relevant information about them according to the schema.

## Core Principles

1. **Extract Explicitly Stated Information Only**: Only extract facts that are directly stated or clearly implied in the input text. Do not infer, assume, or hallucinate information that isn't present. Every piece of extracted data must be traceable back to specific text content.

2. **Follow the Ontology Schema Strictly**: Use only the entity classes and properties defined in the provided ontology schema. Do not create new classes or properties. If information in the text doesn't fit the schema, it's better to omit it than to force it into an inappropriate structure.

3. **Maintain Consistency**: Use consistent entity naming throughout the extraction. If the same entity appears multiple times in the text (even with different names or references), use the exact same entity name in all instances.

## Entity Naming Conventions

**Standard Entities**:
- Use snake_case for multi-word names
- Examples: `alan_turing`, `stanford_university`, `machine_learning_conference`
- Keep names concise but unambiguous
- Use the most canonical or formal name when multiple options exist

**Event Entities**:
- Format: `{{subject}}_{{verb}}_{{object}}_{{YYYY}}`
- Optionally add `_MMDD` for month/day if known
- Use the main subject's canonical name
- Choose a concise, descriptive verb
- Object should be the primary focus, not concatenated details
- Example: `shannon_publishes_information_theory_1948`
- Additional context (location, related documents) should be separate entities with their own properties

**Composite Entities** (for inherent multi-party relationships):
- Format: `{{entity1}}_and_{{entity2}}_{{relationship_type}}`
- Example: `john_doe_and_jane_doe_marriage`
- Use only when the entity inherently involves multiple parties

**Global Uniqueness**: Each entity name must be globally unique and clearly identify exactly one thing. Avoid redundant or concatenated information in names.

## Extraction Process

1. **Identify Entities**: Scan the text for mentions of people, organizations, places, events, documents, concepts, etc. that correspond to classes in the ontology.

2. **Classify Entities**: Determine the appropriate ontology class for each entity. Choose the most specific applicable class.

3. **Extract Properties**: For each entity, extract all property values mentioned in the text according to the ontology schema:
   - **Data Properties**: Literal values (names, dates, numbers, descriptions, etc.)
   - **Object Properties**: References to other entities (use exact entity names)

4. **Resolve References**: Convert pronouns and ambiguous references to specific entity names when the referent is clear from context. If ambiguous, omit rather than guess.

5. **Handle Partial Information**: You can extract partial entity information. Not every property needs to be filled - only include properties for which you have explicit information.

## Incremental Knowledge Building

- **Build Upon Existing Knowledge**: The current knowledge graph state is provided. Build upon it by adding new entities or enriching existing ones with additional information.
- **Enrich Existing Entities**: If you find additional information about entities already in the knowledge graph, you can extract that information by creating a new entity object with the same `name` and `cls` fields. The system will automatically merge the new properties with existing data.
- **Avoid Duplication**: Don't re-extract information already present in the current knowledge graph unless you have new details to add.
- **Consistent Entity References**: When referencing existing entities in object properties, use their exact names as they appear in the current knowledge graph.

## Data Quality Guidelines

- **Avoid Redundancy**: Don't encode the same information in multiple ways. If class hierarchy already encodes a distinction, don't create redundant properties.
- **Respect Property Types**: Ensure data properties contain the correct data types (strings, integers, dates, etc.) and object properties reference valid entity names.
- **Handle Functional Properties**: If a property is functional (can have only one value), ensure you only provide one value per entity.
- **Maintain Open World Assumption**: Missing properties are assumed unknown, not false. Only include properties you have evidence for.

## Output Requirements

Your output must conform exactly to the provided Pydantic schema:
- Each entity must have a `name` field (the entity identifier)
- Each entity must have a `cls` field (the ontology class name)
- Include only properties defined in the ontology for each class
- Use proper data types for all property values
- Reference other entities by their exact names in object properties

## Merging Behavior

When you output multiple entities with the same `name` and `cls` values, the system will automatically merge their properties. This allows you to:
- Add new properties to entities from previous text chunks
- Enrich existing entities with additional details
- Build comprehensive entity profiles incrementally across multiple extractions

## Error Prevention

- **Validate Entity Names**: Ensure all entity names follow naming conventions and are unique
- **Check Property Domains**: Verify that properties are only used with appropriate entity classes
- **Maintain Referential Integrity**: Ensure all entity references in object properties point to actual entities
- **Avoid Schema Violations**: Don't create properties or classes not defined in the ontology

## Context Awareness

- **Coreference Resolution**: Resolve "he", "she", "it", "the company", etc. to specific entities when clear
- **Temporal Context**: Pay attention to dates and time references for proper event naming
- **Spatial Context**: Consider location information for appropriate entity classification and relationships

Remember: Quality over quantity. It's better to extract fewer, highly accurate entities than many entities with questionable information. Focus on creating a clean, consistent knowledge graph that faithfully represents the information in the text according to the ontology schema.
""",
)

# TODO update this as well to be in line with new extraction
# Instructions for ontology-free triplet extraction
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "triplet_extraction_no_ontology",
    f"""
{prompt_registry.tag("triplet_extraction")}
You are tasked with extracting factual (subject, predicate, object) triples from a given input text without any predefined ontology constraints. Extract meaningful relationships and entities based on the content of the text itself.

Extraction Guidelines:

1. Extract Stated Facts Only: Identify only the triples that are explicitly stated in the input text. Do not infer, assume, or add information that the text does not provide. No hallucination or guesswork is allowed - every triple must be directly supported by the text.

2. Include Entity Types (isA): For every unique entity you mention in any triple, include one triple using the predicate isA to state that entity's class/type. Choose appropriate, general class names that describe the entity (e.g., Person, Organization, Location, Event, Concept, etc.). For example: claude_shannon isA Person.

3. Strict Entity Naming Conventions:
    - Use snake_case for multi-word names: Combine words in lowercase with underscores. Examples: alan_turing, vienna_city_hall.
    - Event Entity Format: If the entity represents a specific event or occurrence, name it in the format {{subject}}_{{verb}}_{{object}}_{{YYYY}} (optionally add _MMDD for month and day if known). Use the main subject's canonical name, a concise verb, and an object that is the focus of the event. For example: claude_shannon_develops_phd_dissertation_1939.
    - Compound or Relationship Entities: For composite entities that inherently involve multiple named parties (e.g., a marriage, treaty, or partnership), include the full names of all primary participants to avoid ambiguity. Connect them with _and_ if needed. Example: marriage_john_doe_and_jane_doe.
    - Each entity must have only the information needed to uniquely identify it, and never redundant or concatenated details.

4. Use Meaningful Predicates: Choose predicate names that clearly describe the relationship between entities. Use camelCase for predicates (e.g., hasAuthor, isPartOf, worksAt, founded, etc.). Be consistent with predicate naming throughout the extraction.

5. Consistent Entity References: Maintain consistency in entity naming throughout all triples. If the same entity is mentioned multiple times in the text (even under different names or aliases), use the exact same entity name (same spelling and underscores) every time in your output.

6. Coreference Resolution: Resolve pronouns and ambiguous references in the text to their specific entities. If the text says "He founded the company in 1998" and earlier it's clear that "He" refers to, say, Larry Page, then use the explicit entity name (larry_page) in the triple. Only replace a pronoun with an entity name when you are certain of the reference from the context. If a reference cannot be resolved unambiguously, it's safer to omit that potential triple than to guess.

7. Avoid Overloaded or Redundant Entity Names: Never create entity names that concatenate multiple unrelated elements. Each entity should represent exactly one thing, and additional facts like location, date, or related items should be expressed as separate triples using properties.

## Output Format:

Your final output must be a JSON array (list) of objects, where each object represents one triple. Each object should have exactly three keys: "subject", "predicate", and "object". The values for these keys should be the corresponding entity or literal names (as strings):

- The subject and object should be the entity names following the conventions above (or a literal value if the predicate is assigning an attribute value).
- The predicate should be a meaningful relationship name in camelCase (for isA triples, the predicate is simply "isA").

Format the output as a JSON list [...] containing one object per triple. Do not include any additional commentary or explanation in the output—only the JSON data.

Instructions Recap: Extract all relevant triples from the text, including each entity's isA type triple, and present them as JSON {{subject, predicate, object}} objects. Follow the naming rules and create meaningful relationships. Ensure every fact is backed by the text, with no extraneous or inferred information. Avoid overloaded or redundant entity names. By adhering to these guidelines, the output will consist of high-quality triples ready for knowledge graph construction.
""",
)

# ==================================================#
# ----CQS-------------------------------------------#
# ==================================================#

# Tags
prompt_registry.register_tag(PromptLanguage.ENGLISH, "groups", "GROUPS")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "personas", "PERSONAS")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "questions", "QUESTIONS")
prompt_registry.register_tag(PromptLanguage.ENGLISH, "scope_document", "SCOPE DOCUMENT")

# Instructions
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "generate_groups",
    f"""{prompt_registry.tag("groups")}
You are an ontology engineer in the initial scoping phase of creating a comprehensive ontology for the specified domain.

Your current task is to identify groups of people who possess deep knowledge about this domain. These are NOT people who would help implement or design the ontology itself.

Instead, identify an exhaustive list of domain knowledge holders - the actual experts, practitioners, researchers, users, and other groups who:
- Have first-hand experience with the domain concepts
- Possess specialized knowledge about domain terminology, processes, and relationships
- Work with or use domain-related information in their professional activities
- Can provide insights about what aspects of the domain need to be formalized and understood
- Are likely to have questions or information needs that the ontology should address

Output your response as a properly formatted JSON object with nothing else.""",
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "generate_personas",
    f"""{prompt_registry.tag("personas")}
You are an ontology engineer creating a comprehensive domain ontology. To gather diverse perspectives, you need to interview representative individuals from a specific stakeholder group.

Your task:
Generate exactly the required number of diverse personas from the specified group. Each persona should:
• Represent different experiences, backgrounds, and perspectives relevant to the domain
• Include key characteristics: age, location, education, work experience, and domain-specific knowledge
• Feature relevant personal attributes: interests, technological proficiency, and unique perspectives
• Be described in a natural, detailed manner that highlights their potential contributions to the ontology

Ensure your personas collectively cover the full spectrum of relevant domain experiences and knowledge.
""",
)


prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "deduplicate_questions",
    f"""{prompt_registry.tag("questions")}
Review the provided questions and return only unique questions.

Two questions are duplicates if:
- They ask for the same information (even with different wording)
- They represent the same information need

Instructions:
1. Compare each question against all others
2. Only keep the first occurrence of any semantically identical question (be careful, a more specific question is not a duplicate of a more general one!)
3. Return the deduplicated list in the original format
4. Return ONLY the questions themselves with no additional text

Return only the deduplicated question list and nothing else.
""",
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "generate_scope_document",
    f"""{prompt_registry.tag("scope_document")}
You are a collaborative team of the given personas.

Your task is to create a scope document that defines what is included within the given domain based on the collective expertise of these personas.

## Output Requirements
1. Structure your document with numbered sections and subsections (e.g., 1, 1.1, 1.2)
2. Use bullet points for lists and enumerations
3. Focus on identifying topics, not relationships or processes
4. Do not include any title, introduction, summary, or conclusion - only the content sections

## Content Guidelines
   - Provide a clear, concise definition of the domain
   - Describe the conceptual areas that comprise this domain
   - List all major conceptual areas within the domain
   - For each core topic, list all relevant sub-topics (and go as many levels deep as necessary!)
   - Ensure topics are defined at an appropriate level of abstraction

Remember: Anything mentioned in this document is considered in-scope for the ontology. The document should thoroughly describe what the domain is about.""",
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "merge_scope_documents",
    f"""{prompt_registry.tag("scope_document")}
You are an expert ontology engineer creating an ontology on the given domain.

Your task is to merge the provided scope documents into a single, comprehensive, well-structured document.

## Output Requirements:
1. Structure the document with numbered sections and subsections (e.g., 1, 1.1, 1.1.1)
2. Use bullet points for lists and enumerations
3. Organize content logically from general concepts to specific details

## Content Guidelines:
- Retain all essential information from each source document
- Resolve conflicting information by selecting the most authoritative or comprehensive perspective
- Maintain consistent terminology throughout
- Eliminate redundancies while preserving nuanced differences
- Ensure appropriate technical depth for domain specialists
- Use formal, precise language suitable for technical documentation

## Constraints:
- Do not include information about document authors or personas
- Do not add any external information not present in the source documents
- Do not omit significant details from any source document
- Preserve domain-specific terminology and definitions""",
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "generate_questions",
    f"""{prompt_registry.tag("competency_question")}
You are generating competency questions for an ontology in the specified domain.

## What Are Competency Questions?
Competency questions are specific queries that domain users would want to answer using the ontology. They:
- Represent real information needs of users, not questions about the ontology itself
- Should be answerable using the knowledge captured in the ontology
- Help define the scope and requirements for the ontology

## Examples of Good Competency Questions (for a publication ontology)
- "Who are the authors of paper X?"
- "Which papers cite methodology Y?"
- "What experiments were conducted using equipment Z?"
- "Which publications resulted from grant G?"
- "What are all the research topics covered by lab L?"

## Examples of BAD Questions (These are about the ontology design, not competency questions)
- "How does the ontology represent different types of publications?"
- "How does the ontology model the relationship between authors and papers?"

## Guidelines
1. Write questions from the perspective of domain users, not ontology engineers
2. Focus on specific information users would need to retrieve
3. Ensure questions are concrete and answerable with facts
4. Cover diverse aspects of the domain
5. Phrase questions using natural language as users would ask them
6. Keep most questions concise. If necessary, write two simpler questions instead of one that is too complex in most cases. For 4-5 simple questions, generate a more complex one s.t. we get a good mix!

Generate a list of as many competency questions as required to cover the domain comprehensively.
""",
)
