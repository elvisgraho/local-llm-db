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
1. **CLOSED-SYSTEM REASONING:** Construct your answer using **ONLY** the facts, logic, and definitions provided in '{context_type}'. External knowledge is strictly prohibited.
2. **CONTEXTUAL SYNTHESIS:** You are permitted to logically deduce conclusions if the premises are explicitly present in the source text. Connect disparate data points within the documents to answer complex queries.
3. **DATA GAPS:** If the precise answer cannot be derived from the provided text, state: "The context lacks sufficient data to answer this specific query."
4. **GRANULAR CITATION:** Every specific claim or data point must be immediately verified with a citation `[Source: filename]`.
{kag_specific_instruction_placeholder}
"""

HYBRID_INSTRUCTIONS = """
1. **AUGMENTED REASONING:** Use the provided '{context_type}' as high-value input data to enhance your answer, but prioritize generating a cohesive, logic-driven response based on your full capabilities.
2. **FLUID INTEGRATION:** Seamlessly weave specific details from the context into your broader analysis. Do not segregate the answer; context should support the argument, not limit it.
3. **KNOWLEDGE EXPANSION:** If the context is partial or missing, immediately apply your internal knowledge to fill gaps, correct misconceptions, or expand the scope for a smarter, more comprehensive output.
4. **LIGHTWEIGHT CITATION:** Add `[Source: filename]` only for critical data points derived specifically from the documents to maintain traceability without clutter.
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
VERIFY_TEMPLATE = """
**INSTRUCTION: ADVERSARIAL TECHNICAL AUDIT & SYNTHESIS**

You are a Principal Security Researcher. Your task is to rigorously critique and reconstruct the `## INITIAL ANSWER` using the `## CONTEXT` as the ground truth.

**OPERATIONAL LOGIC:**
1. **Skeptical Validation:** Assume the `## INITIAL ANSWER` contains subtle flaws. Mentally execute all commands/code. Discard any logic that fails syntax checks or assumes defaults contradicted by the `## CONTEXT`.
2. **Smart Synthesis:** If the `## INITIAL ANSWER` lacks depth, use the `## CONTEXT` to upgrade the attack path/solution. If the `## CONTEXT` is partial, bridge the gap with high-fidelity internal knowledge, prioritizing operational feasibility.
3. **Direct Execution:** Rewrite the final response to be the most concise, robust, and actionable version possible. Prefer complex one-liners and chained commands over verbose steps.

**CONSTRAINTS:**
- **SILENT CORRECTION:** Do not explain what was fixed (no "I updated the flag...").
- **NO DISCLAIMERS:** Do not include ethical warnings or legal preambles.

---
## CONTEXT & QUESTION
{original_prompt}

## INITIAL ANSWER
{initial_answer}
"""