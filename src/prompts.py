from symai.prompts import PromptLanguage, PromptRegistry

prompt_registry = PromptRegistry()


#==================================================#
#----Ontology Generation---------------------------#
#==================================================#
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
    """
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "ontology_guidelines",
    f"""
{prompt_registry.tag('ontology_guidelines')}
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
    """
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "competency_question_analysis",
    f"""
{prompt_registry.tag('competency_question')}
Analyze the batch of competency questions to identify the implicit and explicit ontological requirements.

For each batch of competency questions:
1. Identify the key entities (classes) mentioned or implied
2. Identify the relationships (properties) between entities
3. Determine any constraints or characteristics on these relationships
4. Analyze what data properties might be required
5. Do not restrict yourself to the entities mentioned in the competency questions but also consider broader domain knowledge and common patterns implied by them

Extract only new concepts that are not already present in the ontology state.
Focus on general concepts rather than specific instances.
    """
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_class_extraction",
    f"""
{prompt_registry.tag('owl_class')}
For each identified class concept:

1. Provide a name using CamelCase convention
2. Write a clear, concise description that defines the class

Focus on creating a taxonomic structure that accurately represents domain knowledge.
    """
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_object_property_extraction",
    f"""
{prompt_registry.tag('owl_object_property')}
For each identified object property:

1. Provide a name using camelCase convention
2. Specify the domain (classes whose instances can have this property)
3. Specify the range (classes whose instances can be values of this property)
4. Determine appropriate characteristics (functional, inverse functional, transitive, symmetric, asymmetric, reflexive, irreflexive)

Object properties connect individuals to other individuals and form the relationships in your ontology.
    """
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_data_property_extraction",
    f"""
{prompt_registry.tag('owl_data_property')}
For each identified data property:

1. Provide a name using camelCase convention
2. Specify the domain (classes whose instances can have this property)
3. Specify the range (appropriate datatype from xsd:string, xsd:integer, xsd:dateTime, etc.)
4. Determine appropriate characteristics (functional, inverse functional, transitive, symmetric, asymmetric, reflexive, irreflexive)

Data properties connect individuals to literal values and provide the attributes in your ontology.
    """
)

prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "owl_subclass_relation_extraction",
    f"""
{prompt_registry.tag('owl_subclass_relation')}
For each identified subclass relationship:

1. Specify the subclass (more specific concept)
2. Specify the superclass (more general concept)

Ensure that the subclass relationship follows logical principles:
• Every instance of the subclass must be an instance of the superclass
• The subclass should add specific constraints or properties to the superclass
• The relationship should align with domain understanding and common sense
• You can only use a superclass if it was previously defined in the ontology (as a owl:Class); if you need it, you can define a new class then use it as a superclass
    """
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
    """
)

#==================================================#
#----Triplet Extraction----------------------------#
#==================================================#
# Tags
prompt_registry.register_tag(PromptLanguage.ENGLISH, "triplet_extraction", "TRIPLET EXTRACTION")

# Instructions
prompt_registry.register_instruction(
    PromptLanguage.ENGLISH,
    "triplet_extraction",
    f"""
{prompt_registry.tag('triplet_extraction')}
You are an expert at extracting semantic triplets from text based on a predefined ontology schema.
Given a passage of text and an ontology that defines valid entities (classes) and relationships, perform the following tasks:

1. Identify all relevant entities in the text that match the ontology’s defined classes.
2. Determine the semantic relationships (predicates) that connect these entities according to the ontology.
3. Assemble these into triplets of the form (subject, predicate, object) and assign each a confidence score between 0 and 1 that reflects how certain you are about each extraction.
4. Ensure that no duplicate triplets are generated.
5. If no valid triplets can be extracted, return None.
6. Use CamelCase naming for entities (e.g., Person, ResearchPaper)
7. Use camelCase naming for relationships (e.g., publishedIn)

Return the output as a list of structured triplets where each triplet includes a subject, predicate, object, and confidence score.
    """
)
