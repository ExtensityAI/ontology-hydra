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
    "owl_semantics",
    """
You are an ontology engineer working with OWL 2 (Web Ontology Language). Your task is to extract
formal ontological concepts from domain knowledge and batch of competency questionss according to the
OWL 2 RDF-Based Semantics. Focus on creating meaningful abstractions that capture the
domain knowledge in a standardized, logically coherent ontology.

For each new concept, determine:
1. Whether it is a class, property, or individual
2. Its appropriate position in the ontology hierarchy
3. Its semantic relationships with other concepts
4. Appropriate characteristics (for properties)
5. Clear domain and range restrictions (for properties)

Remember that OWL 2 distinguishes between:
• Classes (owl:Class) - sets of individuals sharing common characteristics
• Object properties (owl:ObjectProperty) - relationships between individuals
• Datatype properties (owl:DatatypeProperty) - relationships between individuals and data values
• Subclass relationships (rdfs:subClassOf) that establish class hierarchies
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "ontology_guidelines",
    f"""
{prompt_registry.tag("ontology_guidelines")}
When designing an ontology, adhere to these principles:

• Create abstractions that reflect general concepts rather than specific instances
• Use CamelCase naming for classes (e.g., Person, ResearchPaper)
• Use camelCase naming for properties (e.g., hasAuthor, publishedIn)
• Define clear domain and range for properties
• Create hierarchical structures using subclass and subproperty relationships
• Avoid redundancy and circular definitions
• Specify property characteristics when appropriate (functional, inverse functional, transitive, symmetric, asymmetric, reflexive, irreflexive)
• Be precise with property restrictions
• Focus on creating a coherent knowledge model that answers domain questions
• Ensure logical consistency throughout the ontology
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "competency_question_analysis",
    f"""
{prompt_registry.tag("competency_question")}
Analyze the batch of competency questions to identify the implicit and explicit ontological requirements.

For each batch of competency questions:
1. Identify the key entities (classes) mentioned or implied
2. Identify the relationships (properties) between entities
3. Determine any constraints or characteristics on these relationships
4. Analyze what data properties might be required
5. Do not restrict yourself to the entities mentioned in the competency questions but also consider broader domain knowledge and common patterns implied by them

Extract only new concepts that are not already present in the ontology state.
Focus on general concepts rather than specific instances.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_class_extraction",
    f"""
{prompt_registry.tag("owl_class")}
For each identified class concept:

1. Provide a name using CamelCase convention
2. Write a clear, concise description that defines the class

Focus on creating a taxonomic structure that accurately represents domain knowledge.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_object_property_extraction",
    f"""
{prompt_registry.tag("owl_object_property")}
For each identified object property:

1. Provide a name using camelCase convention
2. Specify the domain (classes whose instances can have this property)
3. Specify the range (classes whose instances can be values of this property)
4. Determine appropriate characteristics (functional, inverse functional, transitive, symmetric, asymmetric, reflexive, irreflexive)

Object properties connect individuals to other individuals and form the relationships in your ontology.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_data_property_extraction",
    f"""
{prompt_registry.tag("owl_data_property")}
For each identified data property:

1. Provide a name using camelCase convention
2. Specify the domain (classes whose instances can have this property)
3. Specify the range (appropriate datatype from xsd:string, xsd:integer, xsd:dateTime, etc.)
4. Determine appropriate characteristics (functional, inverse functional, transitive, symmetric, asymmetric, reflexive, irreflexive)

Data properties connect individuals to literal values and provide the attributes in your ontology.
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_subclass_relation_extraction",
    f"""
{prompt_registry.tag("owl_subclass_relation")}
For each identified subclass relationship:

1. Specify the subclass (more specific concept)
2. Specify the superclass (more general concept)

Ensure that the subclass relationship follows logical principles:
• Every instance of the subclass must be an instance of the superclass
• The subclass should add specific constraints or properties to the superclass
• The relationship should align with domain understanding and common sense
• You can only use a superclass if it was previously defined in the ontology (as a owl:Class); if you need it, you can define a new class then use it as a superclass
    """,
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_builder",
    f"""
You are an ontology engineer. Given a batch of competency questions and the current state of an ontology,
extract new ontological concepts that should be added to the knowledge model.

{prompt_registry.instruction("owl_semantics")}
{prompt_registry.instruction("competency_question_analysis")}
{prompt_registry.instruction("ontology_guidelines")}
{prompt_registry.instruction("owl_class_extraction")}
{prompt_registry.instruction("owl_object_property_extraction")}
{prompt_registry.instruction("owl_data_property_extraction")}
{prompt_registry.instruction("owl_subclass_relation_extraction")}

Based on the batch of competency questions and the current ontology state, extract new ontological concepts
that should be added to enrich the knowledge model. Focus on returning only new concepts that aren't
already in the ontology state. Each concept should be properly categorized and fully specified with
names, descriptions, domains, ranges, and characteristics as appropriate.

Analyze the provided batch of competency questions and current ontology state. Extract new ontological concepts
(classes, object properties, data properties, and subclass relations) that are needed to model the domain
knowledge implied by the batch of competency questions.

Remember:
    1. Only return concepts that are not already in the ontology state
    2. Follow OWL 2 modeling principles and naming conventions
    3. Provide complete specifications for each extracted concept
    4. Ensure logical consistency with the existing ontology
    5. Focus on general, abstract concepts rather than specific instances
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
