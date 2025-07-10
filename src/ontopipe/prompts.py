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

# Instructions
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_builder",
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
* Position appropriately in the class hierarchy
* Ensure each class represents a distinct, coherent domain concept
* Always declare new classes with complete definitions before referencing them in subclass relations
* Each class must be formally defined before it can be used in any relationship
* Create consistent, reusable class definitions that can be referenced multiple times

{prompt_registry.tag("owl_object_property")}
* Connect instances to other instances (relationships between individuals)
* Use camelCase naming starting with a verb (e.g., hasAuthor, isPartOf)
* Specify domain and range classes
* Determine applicable characteristics:
  - Functional: Each subject has at most one value (DO NOT OVERUSE THIS! Reality is often complex. Only use when you are confident!)
  - Inverse functional: Each object relates to at most one subject (DO NOT OVERUSE THIS! Reality is often complex. Only use when you are confident!)
  - Transitive: If A relates to B and B to C, then A relates to C
  - Symmetric: If A relates to B, then B relates to A
  - Asymmetric: If A relates to B, then B cannot relate to A
  - Reflexive: Every entity relates to itself
  - Irreflexive: No entity relates to itself

{prompt_registry.tag("owl_data_property")}
* Connect instances to literal values (attributes)
* Use camelCase naming starting with a verb (e.g., hasTitle, wasPublishedInYear)
* Specify domain classes and appropriate datatype range:
  - xsd:string: Text values (names, titles, descriptions)
  - xsd:integer: Whole numbers (counts, years)
  - xsd:decimal/xsd:float: Numerical values with decimals
  - xsd:dateTime/xsd:date: Date and time values
  - xsd:boolean: True/false values
* Determine if functional (has at most one value per instance)

{prompt_registry.tag("owl_subclass_relation")}
* Establish hierarchical is-a relationships between classes
* Every instance of the subclass must be an instance of the superclass
* Subclass should add specific constraints or properties to the superclass
* Use only previously defined superclasses or define new classes first
* Create logical hierarchies that support inference and query answering

# Modeling Principles
1. Use data properties for literal values, not classes
   - Correct: `publishedInYear` (xsd:integer) data property
   - Incorrect: Creating a `Year` class with object properties

2. Use object properties for relationships, not classes
   - Correct: `hasAuthor` as an object property between `Paper` and `Person`
   - Incorrect: Creating an `Authorship` class to connect papers and people

3. Distinguish between ontology elements and knowledge graph instances
   - Ontology (include): Classes like `Book`, `Adaptation`, `IllustratedEdition`
   - Knowledge graph (exclude): Specific instances like "'Alice in Wonderland'"
   - Focus exclusively on modeling the schema/structure, not specific entities
   - Competency questions may contain specific examples to illustrate queries, but these examples themselves should not become part of the ontology

4. Reuse established ontology design patterns
   - Apply standard solutions before creating custom structures
   - Maintain compatibility with common knowledge modeling approaches

5. Follow minimal ontological commitment
   - Include only concepts essential for answering competency questions
   - Avoid overengineering or excessive detail

6. Ensure logical consistency throughout the ontology
   - Avoid contradictions in class hierarchies and property definitions
   - Maintain coherent semantic relationships

7. Maintain a single-root hierarchical structure
   - Design exactly one top-level abstract class (often named "Thing", but if possible, use a more domain-specific name)
   - Ensure all other classes are descendants (direct or indirect) of this root class
   - Create a coherent tree structure where every class has a path to the root

8. Avoid redundant encoding of information
   - Use subclass relations to encode inherent categorical distinctions
   - Do not create properties that duplicate information already encoded in the class hierarchy
   - Example: If you have Article with subclasses ReviewArticle and EmpiricalStudy, do not create a
     redundant "hasArticleType" data property that encodes this same classification

# Naming Conventions
1. Classes: PascalCase (e.g., `ResearchPaper`, `Author`)
2. Properties: camelCase starting with a verb (e.g., `hasAuthor`, `isPublishedIn`)
3. Create descriptive but concise identifiers
4. Include class names in property names only when necessary for clarity
   - Prefer `hasAuthor` over `hasPaperAuthor` when context is clear
   - Use `isPublishedInJournal` when specificity adds meaning

# Output Requirements
1. Only return concepts not present in the current ontology state
2. For each concept, provide complete specifications:
   - Classes: Name, clear description
   - Properties: Name, domain, range, characteristics
   - Subclass relations: Subclass, superclass
3. Format each concept with appropriate tag for clear identification
4. Focus on general concepts that accurately model the domain knowledge
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
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "triplet_extraction",
    f"""
{prompt_registry.tag("triplet_extraction")}
You are tasked with extracting factual (subject, predicate, object) triples from a given input text, using a provided ontology as reference. The ontology is supplied in JSON format and defines a hierarchy of Classes (with names, descriptions, usage guidelines) as well as Properties—including object properties (relationships between entities) and data properties (attributes or values of entities)—each with specific usage guidelines. Use the ontology to guide what types of entities and relations are valid, and follow all the rules below strictly.

Extraction Guidelines:

1. Extract Stated Facts Only: Identify only the triples that are explicitly stated in the input text. Do not infer, assume, or add information that the text does not provide. No hallucination or guesswork is allowed - every triple must be directly supported by the text.

2. Include Entity Types (isA): For every unique entity you mention in any triple, include one triple using the predicate isA to state that entity’s class/type. The object of this isA triple must be a class name from the ontology that appropriately describes the entity. For example: claude_shannon isA Person. If an entity does not have a corresponding isA triple in your output, that entity is considered invalid. Ensure you choose the correct class from the ontology for each entity.

3. Strict Entity Naming Conventions:
    - Use snake_case for multi-word names: Combine words in lowercase with underscores. Examples: alan_turing, vienna_city_hall.
    - Event Entity Format: If the entity represents a specific event or occurrence, name it in the format {{subject}}_{{verb}}_{{object}}_{{YYYY}} (optionally add _MMDD for month and day if known). Use the main subject's canonical name, a concise verb, and an object that is the focus of the event (not concatenating multiple entities). The object part should be the *main* object relevant to the event—never include more than needed (e.g., do not combine location or year into the object part). For example: claude_shannon_develops_phd_dissertation_1939 for the event where Claude Shannon develops his PhD dissertation in 1939. Any additional information, such as location or associated documents, should be represented as separate entities and attached using properties, not combined in the event entity name.
    - Compound or Relationship Entities: For composite entities that inherently involve multiple named parties (e.g., a marriage, treaty, or partnership), include the full names of all primary participants to avoid ambiguity. Connect them with _and_ if needed. Example: marriage_john_doe_and_jane_doe.
    - Each entity must have only the information needed to uniquely identify it, and never redundant or concatenated details. Do not combine information like location or date in the name unless required by the event pattern above. Names must be globally unique, clear, and concise.

4. Use Ontology-Defined Terms Only: When choosing predicate names (relations) and class names, only use those defined in the provided ontology JSON. Do not invent or assume any new relation or class names that are not in the ontology. Stick exactly to the naming (including capitalization or formatting) of classes and properties as given by the ontology. If the text implies a relationship but the corresponding property is not defined in the ontology, skip that triple.

5. Consistent Entity References: Maintain consistency in entity naming throughout all triples. If the same entity is mentioned multiple times in the text (even under different names or aliases), use the exact same entity name (same spelling and underscores) every time in your output.

6. Coreference Resolution: Resolve pronouns and ambiguous references in the text to their specific entities. If the text says “He founded the company in 1998” and earlier it's clear that “He” refers to, say, Larry Page, then use the explicit entity name (larry_page) in the triple. Only replace a pronoun with an entity name when you are certain of the reference from the context. If a reference cannot be resolved unambiguously, it's safer to omit that potential triple than to guess.

7. Avoid Overloaded or Redundant Entity Names: Never create entity names that concatenate multiple unrelated elements (e.g., including both a document and a location in a single entity name). Each entity should represent exactly one thing, and additional facts like location, date, or related items should be expressed as separate triples using properties. For instance, for an event where Claude Shannon develops his PhD dissertation at Cold Spring Harbor in 1939, represent:
    - The event as claude_shannon_develops_phd_dissertation_1939 (LifeEvent)
    - The dissertation as claude_shannon_phd_dissertation (Document)
    - The location and year as separate properties attached to the event or document, not in the entity name itself.

## Output Format:

Your final output must be a JSON array (list) of objects, where each object represents one triple. Each object should have exactly three keys: "subject", "predicate", and "object". The values for these keys should be the corresponding entity or literal names (as strings):

- The subject and object should be the entity names following the conventions above (or a literal value if the predicate is a data property assigning an attribute value).
- The predicate should be the property name from the ontology (for isA triples, the predicate is simply "isA").

Format the output as a JSON list [...] containing one object per triple. Do not include any additional commentary or explanation in the output—only the JSON data.

### Example Output values:

- {{"subject": "claude_shannon", "predicate": "isA", "object": "Person" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "isA", "object": "LifeEvent" }}
- {{"subject": "claude_shannon_phd_dissertation", "predicate": "isA", "object": "Document" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "has_participant", "object": "claude_shannon" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "happens_in", "object": "1939" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "produces_document", "object": "claude_shannon_phd_dissertation" }}
- {{"subject": "claude_shannon_phd_dissertation", "predicate": "completed_in_year", "object": "1939" }}


Instructions Recap: Extract all relevant triples from the text, including each entity's isA type triple, and present them as JSON {{subject, predicate, object}} objects. Follow the naming rules and use the ontology's vocabulary strictly. Ensure every fact is backed by the text, with no extraneous or inferred information. Avoid overloaded or redundant entity names. By adhering to these guidelines, the output will consist of high-quality triples ready for knowledge graph construction.
""",
)

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

### Example Output values:

- {{"subject": "claude_shannon", "predicate": "isA", "object": "Person" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "isA", "object": "Event" }}
- {{"subject": "claude_shannon_phd_dissertation", "predicate": "isA", "object": "Document" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "hasParticipant", "object": "claude_shannon" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "happensIn", "object": "1939" }}
- {{"subject": "claude_shannon_develops_phd_dissertation_1939", "predicate": "producesDocument", "object": "claude_shannon_phd_dissertation" }}

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

Your current task is to identify groups of people who possess deep knowledge about this domain. These are NOT people who would help implement or design the ontology itself (like developers, ontology engineers, or integration specialists).

Instead, identify an exhaustive list of domain knowledge holders - the actual experts, practitioners, researchers, users, and other groups who:
- Have first-hand experience with the domain concepts
- Possess specialized knowledge about domain terminology, processes, and relationships
- Work with or use domain-related information in their professional activities
- Can provide insights about what aspects of the domain need to be formalized and understood

These domain experts will be interviewed to help scope and define what should be included in the ontology.

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
2. Only keep the first occurrence of any semantically identical question
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
1. Domain Definition:
   - Provide a clear, concise definition of the domain
   - Describe the conceptual areas that comprise this domain

2. Core Topics:
   - List all major conceptual areas within the domain
   - For each core topic, list all relevant sub-topics
   - Ensure topics are defined at an appropriate level of abstraction

3. Terminology:
   - Define domain-specific terms and concepts
   - Identify hierarchical relationships between key concepts

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

Generate a list of as many competency questions as required to cover the domain comprehensively.
""",
)
