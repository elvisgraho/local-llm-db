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
{sources_block_placeholder}

{question_block_placeholder}
"""

# --- 3. INSTRUCTION SETS (Technical Rules Only) ---
STRICT_CONTEXT_INSTRUCTIONS = """
### CORE OPERATING PROTOCOL: EVIDENCE-BASED ENTAILMENT
Your task is to answer the query derived *exclusively* from the provided '{context_type}'.

### INSTRUCTION SET
1. **BOUNDARY DEFINITION (LINGUISTIC VS. FACTUAL):**
   - You MAY use your internal linguistic intelligence to rephrase, structure, and summarize the text for clarity.
   - You MUST NOT introduce external facts, dates, or entities not present in the documents. If a fact is not in the text, it does not exist for this session.

2. **DEDUCTIVE SYNTHESIS (THE "SMART" LAYER):**
   - Perform **Multi-hop Reasoning**: If Document A states "X implies Y" and Document B states "Y implies Z", you must conclude "X implies Z."
   - Explicitly connect disparate data points. Do not just list facts; construct a logical argument based *only* on the provided premises.

3. **STRICT NEGATIVE CONSTRAINTS:**
   - If the specific answer is not explicitly stated or logically entailed by the text, return: "The provided documents do not contain sufficient data to address this inquiry."
   - Do not attempt to guess or hallucinate an answer to please the user.

4. **VERIFIABLE TRACEABILITY:**
   - Append `[Source: filename]` immediately after every distinct claim.
   - Every sentence in your response must be traceable back to a specific sentence in the source text.
"""

HYBRID_INSTRUCTIONS = """
### INSTRUCTION SET
1. **EVALUATE CONTEXT SUFFICIENCY:**
   - First, assess if the provided '{context_type}' is sufficient to answer the query fully. 
   - If the context is partial or noisy, explicitly pivot to your internal knowledge base to fill gaps, labeling these additions as "General Knowledge."

2. **EVIDENCE-BASED SYNTHESIS (NOT SUMMARIZATION):**
   - Do not merely repeat the context. Instead, use the context as *evidence* to support your arguments. 
   - If the context conflicts with established facts or logic, prioritize logical consistency and note the discrepancy.

3. **HYBRID ANSWER GENERATION:**
   - **Step 1 (Internal Reasoning):** Formulate a direct answer based on your general understanding of the topic.
   - **Step 2 (Contextual Validation):** Refine your answer by injecting specific details, figures, or terminology from the provided documents.
   - **Step 3 (Final Polish):** Ensure the tone is authoritative and fluid. Do not segregate "According to the text..." from the rest of the answer.

4. **INTELLIGENT CITATION:**
   - Cite sources `[Source: filename]` only when referencing specific data points (numbers, dates, unique claims). General concepts do not require citation.
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

QUESTION_BLOCK = """
Question: {question}
"""

# --- 5. HELPERS ---
STANDARD_CONTEXT_TYPE = "Context"
DIRECT_TEMPLATE = "Answer based on general knowledge and chat context.\nQuestion: {question}"
VERIFY_TEMPLATE = """
**INSTRUCTION: TECHNICAL QA & ADDITIVE REFINEMENT**

You are a Technical QA Lead. Your goal is to VALIDATE and ENHANCE the `## INITIAL ANSWER` using the `## CONTEXT` as evidence. 

**CORE PHILOSOPHY: NON-DESTRUCTIVE EDITING**
- **Do NOT rewrite** the answer from scratch unless it is fundamentally factually incorrect.
- **Do NOT strip** useful context or reasoning provided in the initial answer.
- **GOAL:** The final output should be the *union* of the Initial Answer's reasoning and the Context's specific facts.

**OPERATIONAL PROTOCOLS:**

1. **FACTUAL INTEGRITY CHECK (The "Filter"):**
   - Scan the `## INITIAL ANSWER` for hallucinations or contradictions against the `## CONTEXT`.
   - *Action:* If a specific claim flatly contradicts the text, **surgically correct** that specific sentence. Do not discard the surrounding logic if it remains valid.

2. **GAP ANALYSIS (The "Supplement"):**
   - Check if the `## CONTEXT` contains high-value details (parameters, specific numbers, edge cases) that were *missed* in the `## INITIAL ANSWER`.
   - *Action:* Weave these missing details into the existing answer to increase depth. Do not replace the answer; **augment it**.

3. **LOGIC REINFORCEMENT:**
   - If the `## INITIAL ANSWER` is vague, use the `## CONTEXT` to make it specific (e.g., replace "run the tool" with the specific command flags found in the docs).
   - If the answer is already excellent, output it almost unchanged, fixing only minor syntax or clarity issues.

**CONSTRAINTS:**
- **PRESERVE TONE:** Maintain the structural format of the Initial Answer (e.g., if it used bullet points, keep them).
- **SILENT IMPROVEMENT:** Output the final polished version directly. No "Here is the corrected version" preambles.

---
## CONTEXT & QUESTION
{original_prompt}

## INITIAL ANSWER
{initial_answer}
"""