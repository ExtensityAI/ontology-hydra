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
  - Functional: Each subject has at most one value
  - Inverse functional: Each object relates to at most one subject
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
   
3. Reuse established ontology design patterns
   - Apply standard solutions before creating custom structures
   - Maintain compatibility with common knowledge modeling approaches
   
4. Follow minimal ontological commitment
   - Include only concepts essential for answering competency questions
   - Avoid overengineering or excessive detail

5. Ensure logical consistency throughout the ontology
   - Avoid contradictions in class hierarchies and property definitions
   - Maintain coherent semantic relationships

6. Maintain a single-root hierarchical structure
   - Design exactly one top-level abstract class (often named "Thing")
   - Ensure all other classes are descendants (direct or indirect) of this root class
   - Create a coherent tree structure where every class has a path to the root

7. Avoid redundant encoding of information
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
1. Analyze the ontology’s subclass relations to detect isolated clusters. Each cluster is a group of classes that are internally connected but not linked to the larger ontology.
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
You are extracting semantic triplets from text to populate a knowledge graph with INSTANCES, not to build the ontology structure itself.

### Task Clarification
- You are NOT defining ontology classes or properties
- You ARE identifying specific instances/entities in the text and their relationships
- The ontology structure (classes and predicates) already exists - use only existing predicates from the ontology

### Key Rules
1. Entity Consistency: Use exact same names for existing entity instances
2. New Entities: Create specific instances with concise, descriptive names (not classes)
3. Predicates: Must be camelCase and exist in the provided ontology (do not create new predicates)
4. Class Assignment: Every entity needs exactly one "isA" triplet ({{"subject": "EntityName", "predicate": "isA", "object": "ExistingClass"}})
5. Classes must already be defined in the ontology - do not create new classes
6. Each entity belongs to exactly one class only - do not assign multiple classes to a single entity

### Extraction Process
1. Identify specific entities/instances in the text (like "John Smith" or "Project Alpha")
2. Determine which existing ontology class each entity belongs to
3. Create exactly one "isA" triplet for each entity to assign its class
4. Extract relationships between the entities using existing predicates
5. Only extract factual relationships explicitly stated or clearly inferable in the text
6. Output should be knowledge graph triplets, not ontology structure definitions""",
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
You are an ontology engineer tasked with creating a comprehensive ontology for the specified domain. 
To ensure your ontology captures all relevant knowledge, perspectives, and use cases, you need to identify 
key stakeholder groups to interview.

Identify an exhaustive list of stakeholder groups who would provide valuable insights for this domain. Output your response as a properly formatted JSON object with nothing else.""",
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
    "generate_questions",
    f"""{prompt_registry.tag("questions")}
You represent a group of different experts as stated below in <group/> tags.. You have been presented with a scope document for an ontology in the given domain.

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
1. Organize your questions into a simple list format (- ...)
2. Ensure each question is specific and concrete
3. Phrase questions to elicit detailed, precise answers
4. ONLY provide the list of questions, nothing else!

Generate only questions that you, as these specific expert personas, would consider relevant and important for understanding the domain. Do not include questions outside your area of expertise.
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

Your task is to create a scope document that defines the key topics and boundaries within the given domain based on the collective expertise of these personas.

## Output Requirements
1. Structure your document with numbered sections and subsections (e.g., 1, 1.1, 1.2)
2. Use bullet points for lists and enumerations
3. Focus on identifying topics, not relationships or processes

## Content Guidelines
* Define what is included in this domain
* Identify what is explicitly excluded
* Note any gray areas or overlaps with adjacent domains

Keep your document concise and focused on establishing a shared vocabulary and clear boundaries for future discussions.""",
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
- Identify and harmonize key ontological elements (classes, properties, relationships, hierarchies)
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
