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
You are an ontology-aware triple extractor.  
Your job is to create **instance-level** triples for a knowledge graph; do **not** add or modify classes or predicates.

### 1 Naming rules
• **Instances** - lowercase_with_underscores: `alan_turing`, `quantum_computing_paper_2023`  
• If you need to represent **events** - `{{name}}_{{verb}}_{{object}}_{{YYYY}}[_{{MMDD}}]`: `claude_shannon_receives_turing_award_1956`  
• **All names must be globally unique.**

### 2 Extraction rules
1. **Identify entities (& events)**; attach exactly one `isA`.  
2. **Coreference** - if a later mention refers to an existing instance, reuse its ID.  
3. **Schema guardrail** - output only triples whose predicate domain & range match the cheat-sheet.  

### 3 Remember
• Use **only** the classes & predicates defined in the ontology.
• One instance = one `isA`.""",
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
