from symai.prompts import PromptLanguage, PromptRegistry

prompt_registry = PromptRegistry()


# ==================================================#
# ----Ontology Generation---------------------------#
# ==================================================#
# Tags
prompt_registry.register_tag(PromptLanguage.ENGLISH, "owl_class", "OWL CLASS")
prompt_registry.register_tag(
    PromptLanguage.ENGLISH, "owl_subclass_relation", "OWL SUBCLASS RELATION"
)
prompt_registry.register_tag(
    PromptLanguage.ENGLISH, "owl_object_property", "OWL OBJECT PROPERTY"
)
prompt_registry.register_tag(
    PromptLanguage.ENGLISH, "owl_data_property", "OWL DATA PROPERTY"
)
prompt_registry.register_tag(
    PromptLanguage.ENGLISH, "competency_question", "COMPETENCY QUESTION"
)
prompt_registry.register_tag(
    PromptLanguage.ENGLISH, "ontology_guidelines", "ONTOLOGY GUIDELINES"
)

# Instructions
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_semantics",
    """
You are an ontology engineer working with OWL 2 (Web Ontology Language). Extract formal ontological concepts from domain knowledge and competency questions according to OWL 2 semantics.

For each concept, determine:
1. Type: Class, property (object or data), or individual
2. Position in the ontology hierarchy
3. Semantic relationships with other concepts
4. Property characteristics and restrictions when applicable

Key OWL 2 components:
• Classes (owl:Class): Sets of individuals sharing common characteristics
• Object properties (owl:ObjectProperty): Relationships between individuals (e.g., authorOf, partOf)
• Datatype properties (owl:DatatypeProperty): Relationships between individuals and literal values (e.g., hasName, hasDate)
• Subclass relationships (rdfs:subClassOf): Hierarchical relationships between classes

Create meaningful abstractions that capture domain knowledge in a standardized, logically coherent ontology.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "ontology_guidelines",
    f"""
{prompt_registry.tag("ontology_guidelines")}
Apply these ontology design principles:

## Fundamental Principles
• Model general concepts rather than specific instances
• Ensure logical consistency throughout the ontology
• Create a coherent knowledge model that answers domain questions

## Naming and Structure
• Classes: Use PascalCase (e.g., Person, ResearchPaper)
• Properties: Use camelCase (e.g., hasAuthor, publishedIn)
• Create hierarchical structures via subclass/subproperty relationships
• Avoid redundancy and circular definitions

## Property Design
• Define clear domain and range for all properties
• Specify appropriate characteristics (functional, transitive, symmetric, etc.)
• Use precise property restrictions to constrain relationships
• Consider inverse properties when relationships are bidirectional
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "competency_question_analysis",
    f"""
{prompt_registry.tag("competency_question")}
Analyze competency questions to identify both explicit and implicit ontological requirements.

For each question:
1. Identify key entities (classes) directly mentioned or implied
2. Extract relationships (properties) between entities
3. Determine constraints, cardinality, or characteristics on relationships
4. Identify attributes (data properties) needed to answer the question
5. Consider domain patterns and broader knowledge structures implied

Extract only new concepts not already present in the ontology state.
Prioritize general concepts (classes, properties) over specific instances.
Focus on concepts necessary to formulate complete answers to the questions.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_class_extraction",
    f"""
{prompt_registry.tag("owl_class")}
For each class concept:

1. Provide a descriptive name using CamelCase convention (e.g., ResearchPaper, ExperimentalMethod)
2. Write a clear, concise definition that establishes its essential characteristics
3. Consider its position in the class hierarchy (what superclasses it might have)
4. Ensure the class represents a distinct, coherent concept within the domain

Focus on creating a taxonomic structure with clear is-a relationships between classes.
Classes should represent categories of things, not attributes or relationships.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_object_property_extraction",
    f"""
{prompt_registry.tag("owl_object_property")}
For each object property:

1. Provide a descriptive name starting with a verb in camelCase (e.g., hasAuthor, isPartOf, collaboratesWith)
2. Specify domain class(es) whose instances can have this property
3. Specify range class(es) whose instances can be values of this property
4. Determine applicable characteristics:
   • Functional: Each subject has at most one value for this property
   • Inverse functional: Each object is related to at most one subject
   • Transitive: If A relates to B and B relates to C, then A relates to C
   • Symmetric: If A relates to B, then B relates to A
   • Asymmetric: If A relates to B, then B cannot relate to A
   • Reflexive: Every entity relates to itself
   • Irreflexive: No entity relates to itself

Object properties connect instances to other instances, forming the relationships in your ontology.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_data_property_extraction",
    f"""
{prompt_registry.tag("owl_data_property")}
For each data property:

1. Provide a descriptive name starting with a verb in camelCase (e.g., hasTitle, wasPublishedInYear, containsText)
2. Specify domain class(es) whose instances can have this property
3. Specify appropriate datatype as range:
   • xsd:string: For text values (names, titles, descriptions)
   • xsd:integer: For whole numbers (counts, years)
   • xsd:decimal/xsd:float: For numerical values with decimals
   • xsd:dateTime/xsd:date: For date and time values
   • xsd:boolean: For true/false values
4. Determine if the property is functional (has at most one value per instance)

Data properties connect instances to literal values, providing attributes for classes in your ontology.
Use data properties for simple attributes rather than creating separate classes.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_subclass_relation_extraction",
    f"""
{prompt_registry.tag("owl_subclass_relation")}
For each subclass relationship:

1. Specify the subclass (more specific concept)
2. Specify the superclass (more general concept)

Ensure each subclass relationship follows these logical principles:
• Every instance of the subclass must be an instance of the superclass (is-a relationship)
• The subclass should add specific constraints or properties to the superclass
• The relationship should align with domain understanding and common sense
• Use only superclasses previously defined in the ontology; if needed, define a new class first

Create a clear, logical hierarchy that supports inference and query answering.
Avoid excessive depth or multiple inheritance unless absolutely necessary.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_builder",
    f"""
You are an ontology engineer extracting concepts from competency questions to enhance an existing ontology.

{prompt_registry.instruction("owl_semantics")}
{prompt_registry.instruction("competency_question_analysis")}
{prompt_registry.instruction("ontology_guidelines")}
{prompt_registry.instruction("owl_class_extraction")}
{prompt_registry.instruction("owl_object_property_extraction")}
{prompt_registry.instruction("owl_data_property_extraction")}
{prompt_registry.instruction("owl_subclass_relation_extraction")}

## Modeling Principles
1. Use data properties for literal values, not classes
   - Correct: `publishedInYear` (xsd:integer) data property 
   - Incorrect: Creating a `Year` class with object properties
   
2. Reuse established ontology design patterns when appropriate
   - Look for standard solutions before creating custom structures
   - Maintain compatibility with common knowledge modeling approaches
   
3. Follow minimal ontological commitment
   - Include only concepts essential for answering competency questions
   - Avoid overengineering or excessive detail

## Naming Conventions
1. Classes: CamelCase with first letter uppercase (e.g., `ResearchPaper`, `Author`)
2. Properties: camelCase starting with a verb (e.g., `hasAuthor`, `isPublishedIn`)
3. Create descriptive but concise identifiers
4. Only include class names in property names when necessary for clarity
   - Prefer `hasAuthor` over `hasPaperAuthor` when context is clear
   - Use `isPublishedInJournal` when specificity adds meaning

## Output Requirements
1. Only return concepts not present in the current ontology state
2. For each concept, provide complete specifications:
   - Classes: Name, clear description
   - Properties: Name, domain, range, characteristics
   - Subclass relations: Subclass, superclass
3. Ensure logical consistency throughout the ontology
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
prompt_registry.register_tag(
    PromptLanguage.ENGLISH, "triplet_extraction", "TRIPLET EXTRACTION"
)

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

Identify a diverse, exhaustive list of stakeholder groups who would provide valuable insights for this domain. Output your response as a properly formatted JSON object with nothing else.""",
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
