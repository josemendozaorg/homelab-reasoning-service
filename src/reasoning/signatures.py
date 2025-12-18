"""DSPy signatures for the reasoning service."""
import dspy

class ReasonSignature(dspy.Signature):
    """You are a precise and direct assistant.

    RULES:
    1. Use <think> tags for internal reasoning.
    2. Check provided 'Context' first. If the answer is in the Context (e.g. Current Date), use it immediately.
    3. If answering, provide the answer text ONLY. Do NOT be verbose.
    
    FORBIDDEN:
    - No meta-commentary (e.g. "The current date is...", "Here is the answer...").
    - No placeholders (e.g. "[insert date]").
    - No self-reference to the context source.
    - No disclaimers about verification.
    
    Example:
    Context: Current Date: 2025-01-01
    Question: What is today's date?
    Response: <think>The date is available in the context.</think>January 1, 2025.
    """
    
    context = dspy.InputField(desc="Relevant context or history", format=str, prefix="Context:\n")
    question = dspy.InputField(desc="The user's question", format=str, prefix="Question:\n")
    response = dspy.OutputField(desc="The model's response (reasoning + answer OR search request)", format=str, prefix="Response:\n")


class CritiqueSignature(dspy.Signature):
    """You are a rigorous critic evaluating reasoning and answers.
    
    Review the provided reasoning and answer for:
    1. Logical errors or gaps
    2. Unsupported assumptions
    3. Missing considerations
    4. Factual inaccuracies
    5. Clarity and completeness
    
    If the answer is completely satisfactory and needs no improvement, respond with exactly:
    Critique: APPROVED
    
    Otherwise, provide specific, actionable feedback.
    """
    
    question = dspy.InputField(desc="The original question")
    reasoning_trace = dspy.InputField(desc="The history of reasoning steps")
    answer = dspy.InputField(desc="The proposed answer")
    critique = dspy.OutputField(desc="Critique or APPROVED", prefix="Critique:\n")


class CritiqueSearchSignature(dspy.Signature):
    """Evaluate search results.
    
    - Do they directly answer the user's question?
    - Are they relevant and sufficient?
    - If they are sufficient, respond with "Search results are sufficient, please formulate an answer."
    - If they are irrelevant or missing info, explain what is missing to guide the next search.
    """
    
    question = dspy.InputField(desc="The original question")
    search_results = dspy.InputField(desc="The results from the web search")
    critique = dspy.OutputField(desc="Evaluation of search results", prefix="Critique:\n")


class RefineSignature(dspy.Signature):
    """RULES:
    1. Take the previous answer and critique.
    2. Produce an improved, cleaner version.
    3. The final answer must be the answer text ONLY.
    
    FORBIDDEN:
    - No "Here is the improved answer"
    - No explanations of what you changed
    - No meta-commentary
    """
    
    question = dspy.InputField(desc="Original question")
    previous_answer = dspy.InputField(desc="The previous answer")
    critique = dspy.InputField(desc="The critique to address")
    improved_response = dspy.OutputField(desc="The improved answer with reasoning")

class SummarizeSignature(dspy.Signature):
    """Summarize the content of a webpage relevant to the query.
    
    Extract key facts, figures, and details from the provided text that are most relevant to the user's query.
    Do not add any external information. If the content is irrelevant, say so.
    """
    
    query = dspy.InputField(desc="The user's original query")
    content = dspy.InputField(desc="The scraped webpage content")
    summary = dspy.OutputField(desc="Concise summary of relevant information")
