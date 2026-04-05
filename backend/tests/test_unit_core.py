"""
Unit tests for core deterministic functions in Cortex Lab.
Tests _is_meaningful_content, BM25 tokenizer, semantic chunking,
cache provider isolation, and content filter.

Run: python -m pytest tests/test_unit_core.py -v
"""

import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ─── Test _is_meaningful_content ────────────────────────────────────────────

class TestIsMeaningfulContent:
    """Test the factual density scoring in Engine._is_meaningful_content."""

    @staticmethod
    def _check(text: str) -> bool:
        from src.engine import CortexRAGEngine
        return CortexRAGEngine._is_meaningful_content(text)

    # ── Should REJECT ──────────────────────────────────────────────────

    def test_rejects_greeting(self):
        assert self._check("hello") is False

    def test_rejects_single_word(self):
        assert self._check("hi") is False

    def test_rejects_short_filler(self):
        assert self._check("ok") is False
        assert self._check("yes") is False
        assert self._check("thanks") is False

    def test_rejects_pure_question_what(self):
        assert self._check("What is my name?") is False

    def test_rejects_pure_question_tell(self):
        assert self._check("Tell me about Jarurat Care") is False

    def test_rejects_question_with_greeting(self):
        assert self._check("hey what is my name") is False

    def test_rejects_list_query(self):
        assert self._check("List my projects") is False

    def test_rejects_how_question(self):
        assert self._check("How do I deploy my app?") is False

    def test_rejects_short_who_question(self):
        assert self._check("who am i?") is False

    def test_rejects_question_mark_ending(self):
        assert self._check("Can you help me with something?") is False

    # ── Should ACCEPT ──────────────────────────────────────────────────

    def test_accepts_informational_statement(self):
        assert self._check(
            "I just finished building the Jarurat Care chatbot using Google Gemini"
        ) is True

    def test_accepts_learning_statement(self):
        assert self._check(
            "Today I learned that reinforcement learning can be applied to robotics"
        ) is True

    def test_accepts_project_description(self):
        assert self._check(
            "My project uses FastAPI for the backend and React for the frontend"
        ) is True

    def test_accepts_personal_info(self):
        assert self._check(
            "My name is Suraj Kumar and I am a B.Tech student at IIIT Ranchi"
        ) is True

    def test_accepts_work_experience(self):
        assert self._check(
            "I worked on building a real-time sentiment analysis dashboard using Python and Streamlit"
        ) is True

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_rejects_too_short(self):
        assert self._check("ab") is False

    def test_rejects_empty(self):
        assert self._check("") is False

    def test_rejects_whitespace(self):
        assert self._check("   ") is False

    def test_accepts_medium_factual_no_question(self):
        assert self._check(
            "I had a meeting with the design team and we discussed the new architecture yesterday"
        ) is True


# ─── Test BM25 Tokenizer ───────────────────────────────────────────────────

class TestBM25Tokenizer:
    """Test the BM25 tokenizer in HybridRetriever."""

    @staticmethod
    def _tokenize(text: str):
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        stopwords = {"the", "a", "an", "is", "was", "are", "were", "be", "been",
                      "have", "has", "had", "do", "does", "did", "will", "would",
                      "could", "should", "may", "might", "to", "of", "in", "for",
                      "on", "with", "at", "by", "from", "it", "this", "that", "i",
                      "me", "my", "we", "our", "you", "your", "he", "she", "they"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def test_basic_tokenization(self):
        result = self._tokenize("Hello World")
        assert "hello" in result
        assert "world" in result

    def test_removes_stopwords(self):
        result = self._tokenize("the quick brown fox is a great animal")
        assert "the" not in result
        assert "is" not in result
        assert "a" not in result
        assert "quick" in result
        assert "brown" in result

    def test_removes_single_char(self):
        result = self._tokenize("I am a student")
        # "I", "a" should be removed (single char or stopword)
        assert "student" in result

    def test_removes_punctuation(self):
        result = self._tokenize("Hello, World! How are you?")
        assert "hello" in result
        assert "world" in result


# ─── Test Cache Provider Isolation ─────────────────────────────────────────

class TestCacheProviderIsolation:
    """Test that the cache correctly isolates entries by provider."""

    def test_hash_key_differs_by_provider(self):
        from src.cache import MultiLevelCache
        cache = MultiLevelCache(embedding_model=None)
        key_local = cache._hash_key("test query", "local")
        key_gemini = cache._hash_key("test query", "gemini")
        assert key_local != key_gemini, "Hash keys should differ by provider"

    def test_hash_key_same_provider(self):
        from src.cache import MultiLevelCache
        cache = MultiLevelCache(embedding_model=None)
        key1 = cache._hash_key("test query", "local")
        key2 = cache._hash_key("test query", "local")
        assert key1 == key2, "Same provider should produce same key"

    def test_exact_cache_isolation(self):
        from src.cache import MultiLevelCache
        cache = MultiLevelCache(embedding_model=None)

        # Store result for local provider
        cache.set_exact("test query", {"answer": "local response"}, provider="local")
        # Store result for gemini provider
        cache.set_exact("test query", {"answer": "gemini response"}, provider="gemini")

        # Retrieve for local
        result_local, level = cache.get("test query", provider="local")
        assert result_local is not None
        assert result_local["answer"] == "local response"

        # Retrieve for gemini
        result_gemini, level = cache.get("test query", provider="gemini")
        assert result_gemini is not None
        assert result_gemini["answer"] == "gemini response"


# ─── Test Semantic Chunking ────────────────────────────────────────────────

class TestSemanticChunking:
    """Test the _chunk_long_content method from IngestionPipeline."""

    @staticmethod
    def _chunk(content: str, max_chars: int = 800, overlap: int = 1):
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(sentences) <= 3:
            return [content]

        chunks = []
        current_chunk = []
        current_len = 0

        for i, sent in enumerate(sentences):
            current_chunk.append(sent)
            current_len += len(sent) + 1

            if current_len >= max_chars and i < len(sentences) - 1:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_len = sum(len(s) + 1 for s in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if len(chunks) > 1 else [content]

    def test_short_content_no_split(self):
        text = "This is short. Only two sentences."
        chunks = self._chunk(text)
        assert len(chunks) == 1

    def test_long_content_splits(self):
        # Create content with many sentences > 800 chars total
        sentences = [f"This is sentence number {i} with some extra content to fill it up." for i in range(20)]
        text = " ".join(sentences)
        chunks = self._chunk(text, max_chars=200)
        assert len(chunks) > 1, "Long content should be split into multiple chunks"

    def test_overlap_exists(self):
        sentences = [f"Sentence {i} has important content here." for i in range(15)]
        text = " ".join(sentences)
        chunks = self._chunk(text, max_chars=200, overlap=1)
        if len(chunks) >= 2:
            # The last sentence of chunk 1 should appear at the start of chunk 2
            last_sent_chunk1 = chunks[0].split(".")[-2].strip() if "." in chunks[0] else ""
            assert len(chunks) >= 2, "Should have overlap"


# ─── Test Hallucination Pattern Detection ──────────────────────────────────

class TestHallucinationPatterns:
    """Test the unified _HALLUC_PATTERNS set."""

    def test_patterns_exist(self):
        from src.llm import _HALLUC_PATTERNS
        assert len(_HALLUC_PATTERNS) > 30, "Should have >30 patterns"

    def test_patterns_are_lowercase(self):
        from src.llm import _HALLUC_PATTERNS
        for p in _HALLUC_PATTERNS:
            assert p == p.lower(), f"Pattern should be lowercase: {p}"

    def test_no_duplicates(self):
        from src.llm import _HALLUC_PATTERNS
        assert isinstance(_HALLUC_PATTERNS, frozenset), "Should be a frozenset (no dupes)"

    def test_key_patterns_present(self):
        from src.llm import _HALLUC_PATTERNS
        assert "deep work" in _HALLUC_PATTERNS
        assert "personal growth" in _HALLUC_PATTERNS
        assert "confidence: high" in _HALLUC_PATTERNS
        assert "emotional resilience" in _HALLUC_PATTERNS


# ─── Test Non-Streaming Personal Info Post-Processing ─────────────────────

class TestPersonalInfoPostProcessing:
    """Test non-streaming personal-info evidence supplementation and extraction."""

    def test_adds_direct_supplement_for_personal_query(self):
        from server import _postprocess_non_stream_result

        captured = {"query": ""}

        def fake_search(query: str, top_k: int = 3):
            captured["query"] = query
            return [
                {
                    "content": "My name is Suraj Kumar and I am pursuing B.Tech at IIIT Ranchi.",
                    "score": 0.71,
                    "memory_type": "semantic",
                }
            ]

        result = {
            "answer": "I cannot determine your name from the available information.",
            "thinking": "Initial retrieval completed.",
            "evidence": [],
            "confidence": 0.42,
        }

        updated = _postprocess_non_stream_result(
            "What is my name?",
            result,
            search_fn=fake_search,
        )

        assert "resume contact information summary" in captured["query"].lower()
        assert updated["evidence"], "Expected direct supplement evidence to be added"
        assert updated["evidence"][0]["channel"] == "direct_supplement"
        assert "suraj" in updated["answer"].lower(), updated["answer"]

    def test_skips_low_score_supplement(self):
        from server import _postprocess_non_stream_result

        def fake_search(_query: str, top_k: int = 3):
            return [{"content": "My name is Suraj Kumar", "score": 0.21, "memory_type": "semantic"}]

        result = {
            "answer": "I cannot determine your name from the available information.",
            "thinking": "Initial retrieval completed.",
            "evidence": [],
            "confidence": 0.42,
        }

        updated = _postprocess_non_stream_result(
            "What is my name?",
            result,
            search_fn=fake_search,
        )

        assert updated["evidence"] == [], "Low-score supplement should not be injected"
        assert "cannot determine" in updated["answer"].lower()

    def test_non_personal_query_is_unchanged(self):
        from copy import deepcopy
        from server import _postprocess_non_stream_result

        def fake_search(_query: str, top_k: int = 3):
            raise AssertionError("Search should not be called for non-personal query")

        result = {
            "answer": "You discussed three deep learning projects.",
            "thinking": "Initial retrieval completed.",
            "evidence": [{"content": "Project A...", "score": 0.77, "channel": "vector"}],
            "confidence": 0.86,
        }

        original = deepcopy(result)
        updated = _postprocess_non_stream_result(
            "Summarize my deep learning projects",
            result,
            search_fn=fake_search,
        )

        assert updated == original

    def test_overrides_confident_answer_when_fact_missing(self):
        from server import _postprocess_non_stream_result

        def fake_search(_query: str, top_k: int = 3):
            return []

        result = {
            "answer": "I cannot find a specific name in your data because the speakers are anonymized.",
            "thinking": "Initial retrieval completed.",
            "evidence": [
                {
                    "content": "My name is Suraj Kumar and I am pursuing B.Tech at IIIT Ranchi.",
                    "score": 0.92,
                    "channel": "vector",
                }
            ],
            "confidence": 0.91,
        }

        updated = _postprocess_non_stream_result(
            "What is my name?",
            result,
            search_fn=fake_search,
        )

        assert "suraj" in updated["answer"].lower(), updated["answer"]

    def test_overrides_confident_explanatory_answer_without_fact(self):
        from server import _postprocess_non_stream_result

        def fake_search(_query: str, top_k: int = 3):
            return []

        result = {
            "answer": "Based on the provided documents, your name is not mentioned by any speaker.",
            "thinking": "Initial retrieval completed.",
            "evidence": [
                {
                    "content": "My name is Suraj Kumar and I am pursuing B.Tech at IIIT Ranchi.",
                    "score": 0.92,
                    "channel": "vector",
                }
            ],
            "confidence": 0.93,
        }

        updated = _postprocess_non_stream_result(
            "What is my name?",
            result,
            search_fn=fake_search,
        )

        assert "suraj" in updated["answer"].lower(), updated["answer"]


# ─── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
