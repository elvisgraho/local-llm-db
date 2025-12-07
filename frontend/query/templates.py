"""
Prompt Templates for Query Processing
"""

# --- 1. SYSTEM PROMPT CONTAINER ---
# We simply take your prompt and append the technical rules.
RAG_SYSTEM_CONSTRUCTION = """{user_defined_persona}

### OPERATIONAL INSTRUCTIONS ###
{rag_instructions}
"""

# --- 2. USER PROMPT (Context + Question) ---
RAG_USER_TEMPLATE = """
{context_block_placeholder}
{relationships_block_placeholder}
{sources_block_placeholder}

{question_block_placeholder}
"""

# --- 3. INSTRUCTION SETS (Technical Rules Only) ---
STRICT_CONTEXT_INSTRUCTIONS = """
1. **SOURCE MATERIAL:** Answer **SOLELY** based on the information explicitly provided in the '{context_type}' section.
2. **NO OUTSIDE KNOWLEDGE:** Do not use external knowledge or training data.
3. **MISSING INFO:** If the information is not in the context, state: "The provided context does not contain enough information."
4. **CITATIONS:** When referencing specific data, cite the source file path like `[Source: filename]`.
{kag_specific_instruction_placeholder}
"""

HYBRID_INSTRUCTIONS = """
1. **PRIORITY:** Base your answer primarily on the provided '{context_type}' section.
2. **CITATIONS:** When referencing the provided documents, cite the source like `[Source: filename]`.
3. **GAPS:** If context is insufficient, explicit state this, then supplement with general knowledge.
4. **DISTINCTION:** Clearly distinguish between what is in the documents and what is general knowledge.
{kag_specific_instruction_placeholder}
"""

# --- 4. DATA BLOCKS ---
CONTEXT_BLOCK = """
{context_type}:
{context}
"""

SOURCES_BLOCK = """
Sources:
{sources}
"""

RELATIONSHIPS_BLOCK = """
Relationships:
{relationships}
"""

QUESTION_BLOCK = """
Question: {question}
"""

# --- 5. HELPERS ---
# (Keeping these for compatibility with processing logic)
KAG_CONTEXT_TYPE = "Knowledge Context"
STANDARD_CONTEXT_TYPE = "Context"
KAG_RELATIONSHIP_QUALIFIER = ", 'Relationships'," 
KAG_RELATIONSHIP_QUALIFIER_CITE = " or 'Relationships'"
KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT = "5. Incorporate specific details from the Knowledge Context directly.\n"
KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID = "5. Incorporate specific details from the Knowledge Context directly.\n"
EMPTY_STRING = ""
DIRECT_TEMPLATE = "Instruction: Answer based on general knowledge.\nQuestion: {question}"