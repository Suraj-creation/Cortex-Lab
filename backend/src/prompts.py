"""
Centralized Prompt Template System for Cortex Lab
All system prompts as versioned templates with injection sanitization.

Replaces scattered hardcoded <|im_start|>...<|im_end|> strings across:
- orchestrator.py
- specialized.py
- query_engine.py
- ingestion/__init__.py
- llm/__init__.py

Usage:
    from src.prompts import PromptBuilder
    prompt = PromptBuilder.faithful_generation(query, evidence_text, session_context)
"""

import re
from typing import List, Optional

# ── Prompt Template Version ──────────────────────────────────────────────────
PROMPT_VERSION = "2.2"

# ── Prompt Injection Sanitization ────────────────────────────────────────────

_INJECTION_MARKERS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>",
    "<|im_sep|>",
    # Common prompt injection attempts
    "ignore previous instructions",
    "ignore all instructions",
    "disregard above",
    "new instructions:",
    "system prompt:",
    "you are now",
]


def sanitize(text: str) -> str:
    """Remove prompt injection markers from user-supplied text.
    Called at the PromptBuilder boundary before any text enters a prompt."""
    if not text:
        return ""
    for marker in _INJECTION_MARKERS:
        text = text.replace(marker, "")
    # Also strip control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


def _format_chat(system: str, user: str, prefix: str = "") -> str:
    """Build a standard ChatML prompt with sanitized user content."""
    user = sanitize(user)
    parts = [
        f"<|im_start|>system\n{system}\n<|im_end|>",
        f"<|im_start|>user\n{user}\n<|im_end|>",
        f"<|im_start|>assistant\n{prefix}",
    ]
    return "\n".join(parts)


# ─── PromptBuilder ───────────────────────────────────────────────────────────

class PromptBuilder:
    """Centralized prompt factory. All LLM prompts flow through here."""

    version = PROMPT_VERSION

    # ── Core Generation ──────────────────────────────────────────────────

    @staticmethod
    def faithful_generation(query: str, evidence_text: str,
                            session_context: str = "") -> str:
        """Stage 1: Grounded generation with inline citations."""
        ctx_line = f"\nSession context: {sanitize(session_context[:200])}" if session_context else ""
        system = (
            "You are Cortex Lab, an intelligent personal AI assistant who knows the user well.\n"
            "Answer their question naturally using the information below.\n\n"
            "RULES:\n"
            "1. Speak conversationally, like a knowledgeable friend — NOT like a database.\n"
            "2. For simple factual questions, answer in one clear sentence.\n"
            "3. For broader questions, write a natural flowing paragraph.\n"
            "4. Always use \"you/your\" when referring to the user, NEVER \"I/my\".\n"
            "5. NEVER say \"Based on your stored memories\" or \"According to evidence\".\n"
            "6. NEVER generate labels like \"Confidence:\", \"Evidence:\", \"Answer:\".\n"
            "7. NEVER generate \"belief evolution\", \"emotion timeline\", \"key insight\", or \"clarity of scope\".\n"
            "8. If the information doesn't contain the answer, say \"I don't have that information yet.\"\n"
            "9. When listing items, include ALL items from the evidence — never truncate lists."
            f"{ctx_line}"
        )
        user = (
            f"{sanitize(query)}\n\n"
            f"Here is what I know about you:\n{sanitize(evidence_text)}"
        )
        return _format_chat(system, user)

    @staticmethod
    def causal_reasoning(query: str, memories_text: str) -> str:
        """Stage 3: Cause-effect chain analysis."""
        system = (
            "You are Cortex Lab's causal reasoning engine. Analyze the user's memories to trace\n"
            "cause-effect relationships. Structure your response as:\n"
            "1. Identify the causal chain (what caused what)\n"
            "2. Note any contributing factors\n"
            "3. Describe the effects/outcomes\n"
            "Be grounded in the evidence — never fabricate causal links."
        )
        user = f"Query: {sanitize(query)}\n\nMemories:\n{sanitize(memories_text)}"
        return _format_chat(system, user)

    @staticmethod
    def self_rag_critique(query: str, answer: str, evidence_text: str) -> str:
        """Stage 4: ISREL/ISSUP/ISUSE critique tokens."""
        system = (
            "You are a retrieval quality evaluator. For the given query, answer, and evidence,\n"
            "evaluate three criteria and provide scores:\n\n"
            "ISREL (Is Relevant): Does the evidence address the query? Score 1-10.\n"
            "ISSUP (Is Supported): Is the answer grounded in the evidence? Score 1-10.\n"
            "ISUSE (Is Useful): Is the answer useful and complete for the user? Score 1-10.\n\n"
            "Output JSON with scores and brief justifications."
        )
        user = (
            f"Query: {sanitize(query)}\n"
            f"Answer: {sanitize(answer[:300])}\n\n"
            f"Evidence:\n{sanitize(evidence_text)}"
        )
        return _format_chat(system, user)

    @staticmethod
    def belief_change(old_text: str, new_text: str, topic: str) -> str:
        """Stage 5: Belief evolution detection."""
        system = (
            "You are a belief evolution detector. Compare two memories about the same topic\n"
            "and classify the change. Types: CONTRADICTION, REFINEMENT, EXPANSION, REINFORCEMENT, NONE.\n"
            "Output JSON with: change_type, old_stance, new_stance, confidence (0-1), explanation."
        )
        user = (
            f"Topic: {sanitize(topic) or 'general'}\n"
            f"Earlier memory: {sanitize(old_text[:300])}\n"
            f"Later memory: {sanitize(new_text[:300])}"
        )
        return _format_chat(system, user)

    @staticmethod
    def raft_generation(query: str, docs_text: str) -> str:
        """Stage 12: Distractor-aware generation (RAFT)."""
        system = (
            "You are Cortex Lab, an intelligent personal AI assistant.\n"
            "Answer the question naturally using ONLY the relevant documents below.\n"
            "Some documents may be irrelevant distractors — ignore them completely.\n\n"
            "RULES:\n"
            "1. Speak conversationally, like a friend — NOT like a database.\n"
            "2. For simple questions, answer in one clear sentence.\n"
            "3. NEVER generate labels like \"Confidence:\", \"Evidence:\", \"Answer:\".\n"
            "4. NEVER generate \"belief evolution\", \"emotion timeline\", or philosophical garbage.\n"
            "5. If no document answers the question, say \"I don't have that information yet.\""
        )
        user = f"{sanitize(query)}\n\nDocuments:\n{sanitize(docs_text)}"
        return _format_chat(system, user)

    @staticmethod
    def function_calling(query: str, tools_desc: str) -> str:
        """Stage 13: Tool/function calling."""
        system = (
            "You are a function calling assistant. Given the user's request and available tools,\n"
            "decide which tool to call and with what arguments.\n"
            'Output JSON: {"tool_name": "...", "arguments": {...}, "reasoning": "..."}\n'
            "If no tool is needed, set tool_name to \"none\".\n\n"
            f"Available tools:\n{tools_desc}"
        )
        user = sanitize(query)
        return _format_chat(system, user)

    # ── Routing & Query Intelligence ─────────────────────────────────────

    @staticmethod
    def route_query(query: str, session_context: str = "") -> str:
        """Stage 2: Structured JSON intent classification."""
        system = (
            "You are Cortex Lab's query router. Analyze the user's query and output a JSON routing decision.\n\n"
            "Available intents: temporal, causal, reflective, factual, procedural, comparative, exploratory\n"
            "Available agents: timeline, causal, reflection, planning, arbitration\n"
            "Complexity: low (0.0-0.3), medium (0.3-0.6), high (0.6-1.0)"
        )
        ctx = f"Session context: {sanitize(session_context[:200])}\n" if session_context else ""
        user = f"{ctx}Query: {sanitize(query)}\n\nOutput routing decision as JSON:"
        return _format_chat(system, user)

    @staticmethod
    def multi_query_generation(query: str) -> str:
        """RAG-Fusion multi-query generation."""
        system = (
            "Generate 3 different versions of the following question.\n"
            "Each version MUST ask about the same topic and preserve the original meaning.\n"
            "Only rephrase — do NOT change the subject or introduce new topics.\n"
            "Output one version per line, numbered 1-3."
        )
        user = sanitize(query)
        return _format_chat(system, user, prefix="1.")

    @staticmethod
    def hyde_generation(query: str) -> str:
        """HyDE: Hypothetical Document Embedding."""
        system = (
            "Write a brief hypothetical answer (2-3 sentences) to this question,\n"
            "as if answering from personal memories."
        )
        return _format_chat(system, sanitize(query))

    @staticmethod
    def step_back_generation(query: str) -> str:
        """Step-back prompting: generate more abstract question."""
        system = (
            "Given this specific question, generate ONE more general question\n"
            "that would provide useful background context."
        )
        return _format_chat(system, sanitize(query))

    @staticmethod
    def query_decomposition(query: str) -> str:
        """Decompose complex query into sub-queries."""
        system = (
            "Break this complex question into 2-3 simpler sub-questions\n"
            "that can each be answered independently.\n"
            "Output one sub-question per line, numbered 1-3."
        )
        return _format_chat(system, sanitize(query), prefix="1.")

    # ── Agent-Specific Prompts ───────────────────────────────────────────

    @staticmethod
    def no_retrieval(query: str) -> str:
        """Direct answer for simple queries (no memory retrieval)."""
        system = (
            "You are Cortex Lab, a personal AI memory and reasoning assistant.\n"
            "If this is a personal question about the user and you don't have stored memories\n"
            "about it, honestly say you don't have that information yet.\n"
            "Never fabricate personal details."
        )
        return _format_chat(system, sanitize(query))

    @staticmethod
    def timeline_no_evidence(query: str) -> str:
        """Timeline agent fallback when no memories found."""
        system = (
            "You are Cortex Lab, an AI memory assistant. The user asked about a timeline\n"
            "but no relevant memories were found. Say so honestly."
        )
        return _format_chat(system, sanitize(query))

    @staticmethod
    def multi_step_synthesis(query: str, combined_answers: str) -> str:
        """Multi-agent synthesis for complex queries."""
        system = (
            "You are Cortex Lab, synthesizing multi-agent analysis of the user's memories.\n"
            "Be concise but thorough. If no relevant memories exist, say so honestly."
        )
        user = f"{sanitize(query)}\n\nAgent Analyses:\n{sanitize(combined_answers)}"
        return _format_chat(system, user)

    @staticmethod
    def self_rag_revision(query: str, answer: str, evidence_text: str,
                          weak_area: str) -> str:
        """Self-RAG revision prompt targeting a specific weakness."""
        system = f"Revise this answer to improve {weak_area}. Be grounded in the evidence."
        user = (
            f"Question: {sanitize(query)}\n"
            f"Original answer: {sanitize(answer[:300])}\n"
            f"Evidence: {sanitize(evidence_text)}\n\n"
            f"Improved answer (focus on {weak_area}):"
        )
        return _format_chat(system, user)

    @staticmethod
    def arbitration(query: str, evidence_text: str) -> str:
        """Arbitration agent: conflict resolution with citations."""
        system = (
            "You are Cortex Lab, an intelligent personal AI assistant.\n"
            "Analyze the evidence for contradictions and conflicting information.\n"
            "Determine which is most likely correct based on recency, confidence, and context.\n"
            "Explain the evolution from old belief to new belief.\n"
            "Cite evidence with [1], [2], etc."
        )
        user = f"{sanitize(query)}\n\nEvidence:\n{sanitize(evidence_text)}"
        return _format_chat(system, user)

    # ── Ingestion Prompts ────────────────────────────────────────────────

    @staticmethod
    def classify_memory_type(text: str) -> str:
        """Memory type classification fallback (LLM)."""
        system = "Classify this memory into one type."
        user = (
            f"\"{sanitize(text[:200])}\"\n\n"
            "Types: episodic (events/activities), semantic (facts/knowledge), "
            "procedural (processes/how-to), reflective (thoughts/realizations)"
        )
        return _format_chat(system, user)

    @staticmethod
    def proposition_extraction(text: str) -> str:
        """Atomic proposition decomposition (EMNLP 2024)."""
        system = (
            "Decompose text into independent atomic facts.\n"
            "Each fact must be self-contained. One fact per line. No numbering."
        )
        return _format_chat(system, sanitize(text[:500]))

    @staticmethod
    def context_prefix(content: str, session_context: str) -> str:
        """Generate contextual prefix for a memory (Anthropic-style)."""
        system = (
            "Write a SHORT context (1-2 sentences) to situate this memory.\n"
            "Include who/what/when if relevant."
        )
        user = (
            f"Session context:\n{sanitize(session_context[:500])}\n\n"
            f"Memory:\n{sanitize(content[:300])}"
        )
        return _format_chat(system, user)

    @staticmethod
    def greeting_response(user_message: str) -> str:
        """Casual greeting/conversation response (used in server.py streaming)."""
        system = (
            "You are Cortex Lab, a friendly and warm personal AI assistant.\n"
            "The user is greeting you or making casual conversation. Respond naturally and briefly.\n"
            "Be cheerful, helpful, and personable. Keep it to 1-2 sentences.\n"
            "Do NOT reference memories, evidence, or past conversations.\n"
            "Do NOT generate philosophical content, analysis, or \"key insights\"."
        )
        return _format_chat(system, sanitize(user_message))

    @staticmethod
    def pageindex_generation(user_message: str, evidence_block: str) -> str:
        """Document-aware prompt for PageIndex evidence (uploaded documents)."""
        system = (
            "You are Cortex Lab, an intelligent AI assistant with access to the user's uploaded documents and memories.\n"
            "The user is asking about content from their uploaded documents. The relevant document content is provided below.\n\n"
            "RULES:\n"
            "- Answer ONLY based on the document content provided below\n"
            "- Be thorough and detailed — include specific facts, names, and numbers from the documents\n"
            "- Organize your answer clearly with bullet points or sections if the content covers multiple topics\n"
            "- If the documents don't contain the answer, say \"I couldn't find that in your uploaded documents\"\n"
            "- NEVER make up information that isn't in the provided content\n"
            "- NEVER add citations like [1] [2] — just speak naturally\n"
            "- NEVER generate \"Confidence:\", \"Evidence:\", \"Answer:\" labels"
        )
        user = (
            f"{sanitize(user_message)}\n\n"
            f"Here is the relevant content from your uploaded documents:\n{evidence_block}"
        )
        return _format_chat(system, user)

    @staticmethod
    def streaming_rag_generation(user_message: str, evidence_block: str) -> str:
        """Standard RAG prompt for streaming responses (personal memories)."""
        system = (
            "You are Cortex Lab, an intelligent personal AI assistant who knows the user well.\n"
            "You have access to the user's stored memories below. Use them to answer naturally.\n\n"
            "PERSONALITY:\n"
            "- Speak warmly and conversationally, like a knowledgeable friend\n"
            "- Give direct, confident answers — never say \"Based on your stored memories\" or \"According to evidence\"\n"
            "- For simple questions (name, email, location), answer in ONE short sentence\n"
            "- For broader questions (skills, projects, background), write a flowing natural paragraph — NOT bullet lists\n"
            "- Always use \"you/your\" when referring to the user, NEVER \"I/my\" (those are the USER's facts, not yours)\n"
            "- NEVER add citations like [1] [2] — just speak naturally\n"
            "- NEVER generate \"Confidence:\", \"Evidence:\", \"Answer:\" labels\n"
            "- NEVER say \"belief evolution\", \"emotion timeline\", \"key insight\", \"clarity of scope\", or similar generic phrases\n\n"
            "If the evidence doesn't answer the question, simply say \"I don't have that information yet — feel free to tell me and I'll remember it!\""
        )
        user = (
            f"{sanitize(user_message)}\n\n"
            f"Here is what I know about you:\n{evidence_block}"
        )
        return _format_chat(system, user)

    @staticmethod
    def synthesis_rag_generation(user_message: str, evidence_block: str) -> str:
        """Synthesis/comprehensive RAG prompt for complex, vision, and philosophical queries."""
        system = (
            "You are Cortex Lab, an intelligent personal AI assistant who deeply understands the user.\n"
            "The user is asking a synthesis question that requires a comprehensive, thoughtful answer.\n"
            "You have extensive stored memories and knowledge about them below.\n\n"
            "RESPONSE GUIDELINES:\n"
            "- Write a THOROUGH, multi-paragraph response that covers ALL relevant aspects from the evidence\n"
            "- Weave together themes, ideas, and details into a cohesive narrative\n"
            "- Be specific — reference concrete projects, ideas, writings, and experiences from the evidence\n"
            "- Connect different pieces of evidence to paint a complete picture\n"
            "- Write at least 3-5 paragraphs for complex questions about vision, philosophy, or worldview\n"
            "- Speak warmly and conversationally, like a deeply knowledgeable friend\n"
            "- Always use \"you/your\" when referring to the user\n"
            "- NEVER truncate your answer — finish every thought completely\n"
            "- NEVER add citations like [1] [2] — just speak naturally\n"
            "- NEVER generate \"Confidence:\", \"Evidence:\", \"Answer:\" labels\n"
            "- NEVER say \"Based on stored memories\" or similar meta-commentary\n\n"
            "If the evidence doesn't fully answer the question, share what you do know and note what's missing."
        )
        user = (
            f"{sanitize(user_message)}\n\n"
            f"Here is what I know about you:\n{evidence_block}"
        )
        return _format_chat(system, user)

    # ── RAPTOR ───────────────────────────────────────────────────────────

    @staticmethod
    def raptor_summary(combined_texts: str) -> str:
        """Generate a RAPTOR cluster summary from a group of related memories."""
        system = (
            "You are a summarization assistant. Given a cluster of related memories,\n"
            "write a concise 2-3 sentence summary that captures the key themes,\n"
            "facts, and relationships across all the memories.\n"
            "Be specific — include names, projects, and topics mentioned.\n"
            "Do NOT add commentary or analysis."
        )
        return _format_chat(system, sanitize(combined_texts))

    # ── Community Summary ────────────────────────────────────────────────

    @staticmethod
    def community_summary(community_members: str, sample_content: str) -> str:
        """Generate a summary for a graph community cluster."""
        system = (
            "Summarize this group of related entities and their connections in 2-3 sentences.\n"
            "Include the key entities, their relationships, and what they have in common."
        )
        user = f"Entities: {sanitize(community_members)}\n\nSample content:\n{sanitize(sample_content[:500])}"
        return _format_chat(system, user)
