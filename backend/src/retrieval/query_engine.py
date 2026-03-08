"""
Query Intelligence Layer for Cortex Lab
- Intent Detection (keyword heuristics + LLM fallback via route_query Stage 2)
- Complexity Scoring
- Adaptive Routing
- Multi-Query Generation (RAG-Fusion)
- HyDE (Hypothetical Document Embedding)
- Step-Back Prompting
- Query Decomposition
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.models import MemoryQuery, QueryIntent, RoutingStrategy
from src.models.embeddings import EmbeddingModel
from src.llm import LocalLLM
from src.prompts import PromptBuilder, sanitize


class QueryAnalyzer:
    """
    Analyzes queries to determine intent, complexity, and routing.
    Uses keyword heuristics first, with LLM fallback for ambiguous cases.
    The LLM fallback leverages route_query (Stage 2 fine-tuning) but is
    deferred to the orchestrator for async execution.
    """

    # Intent keyword mappings (ordered by specificity — more specific intents first)
    INTENT_KEYWORDS = {
        QueryIntent.TEMPORAL: [
            "when", "what time", "how long ago", "last week", "yesterday",
            "last month", "in january", "in february", "in march", "in april",
            "in may", "in june", "in july", "in august", "in september",
            "in october", "in november", "in december", "timeline", "chronolog",
            "sequence", "before", "after", "during",
        ],
        QueryIntent.CAUSAL: [
            "why", "because", "caused", "led to", "reason", "result of",
            "consequence", "what made me", "what caused", "how come",
            "factor", "influence",
        ],
        QueryIntent.REFLECTIVE: [
            "how did my", "changed", "evolved", "pattern", "realized",
            "over time", "growth", "progress", "trend", "shift in",
            "belief", "opinion changed",
            "vision", "dream", "philosophy", "paradigm", "worldview",
            "aspiration", "ideology", "core belief", "values",
            "reimagining", "redefining", "rethinking", "transforming",
            "perspective on", "approach to education",
        ],
        QueryIntent.PROCEDURAL: [
            "how do", "how to", "steps", "process", "method",
            "procedure", "workflow", "guide", "my process",
            "review process", "my method", "my workflow",
        ],
        QueryIntent.FACTUAL: [
            "what is", "what are", "who is", "define", "explain",
            "describe", "what did i learn",
        ],
        QueryIntent.COMPARATIVE: [
            "compare", "difference", "similar", "versus", "vs",
            "better", "worse", "prefer",
        ],
        QueryIntent.EXPLORATORY: [
            "tell me", "what about", "anything about", "related to",
            "show me", "find", "search",
        ],
    }

    # Complexity indicators
    COMPLEXITY_BOOSTERS = [
        "why", "how did", "evolution", "over time", "relationship between",
        "compare", "analyze", "pattern", "all the times", "chain of events",
        "led to", "caused", "history of", "trace",
        "everything about", "all about", "tell me about", "what do you know about",
        "summarize", "comprehensive", "in detail", "elaborate",
        "vision", "dream", "philosophy", "paradigm", "worldview",
        "aspiration", "core belief", "core vision", "ideology",
        "reimagining", "redefining", "rethinking", "transforming",
        "fundamental", "deeply", "perspective",
    ]

    def analyze(self, query: str) -> MemoryQuery:
        """Full query analysis: intent + complexity + routing + temporal extraction."""
        t0 = time.time()
        query_lower = query.lower().strip()

        # 0. Fast path: detect trivial/greeting queries → skip retrieval entirely
        if self._is_greeting_or_trivial(query_lower):
            result = MemoryQuery(
                raw_query=query,
                intent=QueryIntent.EXPLORATORY,
                complexity=0.0,
                routing=RoutingStrategy.NO_RETRIEVAL,
                confidence=0.95,
            )
            elapsed = (time.time() - t0) * 1000
            print(f"  🔍 Query analyzed: GREETING/TRIVIAL → NO_RETRIEVAL ({elapsed:.0f}ms)")
            return result

        # 1. Detect intent
        intent = self._detect_intent(query_lower)

        # 2. Score complexity (intent-aware)
        complexity = self._score_complexity(query_lower, intent)

        # 3. Determine routing
        routing = self._determine_routing(complexity)

        # 4. Extract temporal constraints
        time_start, time_end = self._extract_temporal(query_lower)

        # 5. Extract entities from query
        entities = self._extract_query_entities(query)

        # 6. Extract topics
        topics = self._extract_query_topics(query_lower)

        result = MemoryQuery(
            raw_query=query,
            intent=intent,
            complexity=complexity,
            routing=routing,
            time_start=time_start,
            time_end=time_end,
            entities=entities,
            topics=topics,
            confidence=0.8,
        )

        elapsed = (time.time() - t0) * 1000
        print(f"  🔍 Query analyzed: intent={intent.value}, complexity={complexity:.2f}, routing={routing.value} ({elapsed:.0f}ms)")

        return result

    @staticmethod
    def _is_greeting_or_trivial(query: str) -> bool:
        """Detect greetings, filler, and trivial messages that don't need retrieval."""
        q = query.strip().lower().rstrip("!?.,;:")
        words = q.split()

        # Very short messages (1-3 words)
        if len(words) <= 3:
            greeting_words = {
                "hi", "hey", "hello", "hii", "hiii", "yo", "sup", "howdy",
                "hola", "namaste", "bonjour", "helo", "heyyy",
                "good", "morning", "evening", "afternoon", "night",
                "thanks", "thank", "you", "ok", "okay", "bye", "goodbye",
                "yes", "no", "yeah", "nah", "sure", "hmm", "hm",
                "how", "are", "doing", "there", "whats", "up",
                "welcome", "great", "nice", "cool", "awesome",
            }
            if all(w in greeting_words for w in words):
                return True

        # Common greeting patterns
        greeting_patterns = [
            "hi", "hey", "hello", "hii", "hiii", "yo", "sup", "howdy",
            "good morning", "good evening", "good night", "good afternoon",
            "how are you", "how's it going", "what's up", "whats up",
            "nice to meet", "how do you do", "hey there", "hello there",
            "hi there", "thanks", "thank you", "bye", "goodbye", "see you",
            "ok", "okay", "sure", "great", "nice", "cool", "awesome",
        ]
        if q in greeting_patterns:
            return True
        # Also match with trailing punctuation stripped
        for pat in greeting_patterns:
            if q == pat or q.startswith(pat + " ") and len(q) < len(pat) + 10:
                return True

        return False

    def _detect_intent(self, query: str) -> QueryIntent:
        """Keyword-based intent detection with weighted scoring.
        More specific intents (PROCEDURAL, CAUSAL) get a small priority boost
        to break ties with broader intents (FACTUAL, EXPLORATORY)."""
        scores = {}
        # Priority weights: specific intents get a slight tiebreaker boost
        priority = {
            QueryIntent.TEMPORAL: 0.01,
            QueryIntent.CAUSAL: 0.02,
            QueryIntent.REFLECTIVE: 0.02,
            QueryIntent.PROCEDURAL: 0.03,  # Higher priority over FACTUAL
            QueryIntent.FACTUAL: 0.0,
            QueryIntent.COMPARATIVE: 0.02,
            QueryIntent.EXPLORATORY: -0.01,  # Lowest priority (catch-all)
        }
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                scores[intent] = score + priority.get(intent, 0)

        if scores:
            # If PROCEDURAL and FACTUAL both match, PROCEDURAL wins via priority
            return max(scores, key=scores.get)
        return QueryIntent.EXPLORATORY

    def _score_complexity(self, query: str, intent: QueryIntent = None) -> float:
        """Score query complexity 0.0-1.0.
        Intent-aware: reflective/comparative queries get a complexity boost."""
        score = 0.3  # baseline

        # Word count (higher thresholds for stronger boost)
        words = len(query.split())
        if words > 10:
            score += 0.05
        if words > 15:
            score += 0.10
        if words > 25:
            score += 0.10
        if words > 40:
            score += 0.05

        # Complexity indicators
        for booster in self.COMPLEXITY_BOOSTERS:
            if booster in query:
                score += 0.1

        # Multiple questions
        if query.count("?") > 1:
            score += 0.15

        # Conjunctions suggesting multi-part
        if any(w in query for w in ["and", "also", "additionally", "then"]):
            score += 0.05

        # Intent-based complexity boost (reflective/comparative inherently complex)
        if intent in (QueryIntent.REFLECTIVE, QueryIntent.COMPARATIVE):
            score += 0.20
        elif intent == QueryIntent.CAUSAL:
            score += 0.10

        return min(score, 1.0)

    def _determine_routing(self, complexity: float) -> RoutingStrategy:
        """Determine routing strategy based on complexity."""
        if complexity < 0.3:
            return RoutingStrategy.NO_RETRIEVAL
        elif complexity < 0.6:
            return RoutingStrategy.SINGLE_STEP
        else:
            return RoutingStrategy.MULTI_STEP

    def _extract_temporal(self, query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract time range from query. Handles both relative and absolute dates."""
        now = datetime.now()
        start = None
        end = None

        # Relative time patterns
        if "yesterday" in query:
            start = now - timedelta(days=1)
            end = now
        elif "last week" in query:
            start = now - timedelta(weeks=1)
            end = now
        elif "last month" in query:
            start = now - timedelta(days=30)
            end = now
        elif "last year" in query:
            start = now - timedelta(days=365)
            end = now
        elif "today" in query:
            start = now.replace(hour=0, minute=0, second=0)
            end = now

        # Month name patterns (with optional day: "March 15", "March 15th")
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        for month_name, month_num in months.items():
            if month_name in query:
                year = now.year
                # If month is in the future, use last year
                if month_num > now.month:
                    year -= 1

                # Try to extract day number: "March 15", "March 15th"
                day_match = re.search(
                    rf'{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?',
                    query, re.IGNORECASE
                )
                if day_match:
                    day = min(int(day_match.group(1)), 28)  # Safe cap
                    start = datetime(year, month_num, day)
                    end = start + timedelta(days=1)
                else:
                    start = datetime(year, month_num, 1)
                    if month_num == 12:
                        end = datetime(year + 1, 1, 1)
                    else:
                        end = datetime(year, month_num + 1, 1)
                break

        # Quarter patterns: "Q1 2024", "Q3 2025"
        quarter_match = re.search(r'q([1-4])\s*(\d{4})', query, re.IGNORECASE)
        if quarter_match:
            q = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            start_month = (q - 1) * 3 + 1
            start = datetime(year, start_month, 1)
            end_month = start_month + 3
            end_year = year
            if end_month > 12:
                end_month = 1
                end_year = year + 1
            end = datetime(end_year, end_month, 1)

        # Bare year pattern: "in 2023", "2024", "during 2025"
        if start is None:
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                year = int(year_match.group(1))
                if 2000 <= year <= now.year + 1:
                    start = datetime(year, 1, 1)
                    end = datetime(year + 1, 1, 1)

        return start, end

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity references from query.
        Handles possessives (TechCorp's → TechCorp) and multi-word entities."""
        entities = []

        # Multi-word capitalized sequences (e.g. "Cortex Lab", "Deep Learning")
        multi_word = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', query)
        for mw in multi_word:
            clean_mw = re.sub(r"[''']s$", "", mw).strip()
            if len(clean_mw) > 2:
                entities.append(clean_mw)

        # Single capitalized words
        words = query.split()
        for word in words:
            # Strip possessives before cleaning
            word = re.sub(r"[''']s$", "", word)
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 1:
                # Skip if already captured as part of multi-word entity
                if not any(clean in mw for mw in entities):
                    entities.append(clean)
        return entities

    def _extract_query_topics(self, query: str) -> List[str]:
        """Extract topics from query."""
        topic_map = {
            "work": ["work", "job", "career", "office", "project"],
            "health": ["health", "exercise", "gym", "doctor", "sleep"],
            "relationships": ["friend", "family", "partner", "relationship"],
            "learning": ["learn", "study", "course", "book", "research"],
            "technology": ["code", "programming", "AI", "machine learning"],
            "finance": ["money", "budget", "invest", "salary"],
        }
        topics = []
        for topic, keywords in topic_map.items():
            if any(kw in query for kw in keywords):
                topics.append(topic)
        return topics


class QueryTransformer:
    """
    Transforms queries for improved retrieval coverage.
    - Multi-Query Generation (RAG-Fusion)
    - HyDE (Hypothetical Document Embedding)
    - Step-Back Prompting
    - Query Decomposition
    """

    def __init__(self, llm: LocalLLM, embedding_model: EmbeddingModel):
        self.llm = llm
        self.embeddings = embedding_model

    def transform(self, query: MemoryQuery) -> MemoryQuery:
        """Apply all relevant transformations based on routing strategy.
        Uses a single batched LLM call when Gemini is the backend (§2.1)."""
        t0 = time.time()

        if query.routing == RoutingStrategy.NO_RETRIEVAL:
            return query  # Skip transformations for simple queries

        # ── Detect if Gemini API is the LLM backend ──────────────────────
        # For Gemini: use a single batched call to generate all variants at once.
        # For local model: use parallel thread pool (original behavior).
        using_gemini_llm = getattr(self.llm, 'model', None) == 'gemini-api'

        if using_gemini_llm:
            self._transform_batched_gemini(query)
        else:
            self._transform_parallel_local(query)

        # Generate query embedding (1 API call)
        query.embedding = self.embeddings.embed(query.raw_query).tolist()

        elapsed = (time.time() - t0) * 1000
        variants = (len(query.multi_queries) + (1 if query.hyde_answer else 0)
                    + (1 if query.step_back_query else 0) + len(query.sub_queries))
        print(f"  🔄 Query transformed: {variants} variants ({elapsed:.0f}ms)")
        return query

    def _transform_batched_gemini(self, query: MemoryQuery):
        """Single Gemini LLM call that produces ALL query variants at once.
        Reduces 3 separate API calls → 1 API call per query."""
        need_hyde = query.intent in (QueryIntent.FACTUAL, QueryIntent.EXPLORATORY, QueryIntent.PROCEDURAL)
        need_stepback = (query.intent in (QueryIntent.CAUSAL, QueryIntent.REFLECTIVE)
                         and query.complexity > 0.5)
        need_decompose = query.routing == RoutingStrategy.MULTI_STEP

        # Always start with multi-queries via entity detection (fast, no LLM)
        query.multi_queries = self._generate_multi_queries(query.raw_query)

        # For simple factual queries: just multi-queries + HyDE in one LLM call
        if need_hyde and not need_stepback and not need_decompose:
            hyde = self._generate_hyde(query.raw_query)
            if hyde:
                query.hyde_answer = hyde
            return

        # Skip optional transformations for API queries to save quota
        # (multi-queries alone are sufficient for good retrieval)

    def _transform_parallel_local(self, query: MemoryQuery):
        """Parallel query transformation using ThreadPoolExecutor for local LLM."""
        from concurrent.futures import ThreadPoolExecutor
        futures = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="qtransform") as pool:
            futures["multi_queries"] = pool.submit(self._generate_multi_queries, query.raw_query)
            if query.intent in (QueryIntent.FACTUAL, QueryIntent.EXPLORATORY, QueryIntent.PROCEDURAL):
                futures["hyde"] = pool.submit(self._generate_hyde, query.raw_query)
            if query.intent in (QueryIntent.CAUSAL, QueryIntent.REFLECTIVE) and query.complexity > 0.5:
                futures["step_back"] = pool.submit(self._generate_step_back, query.raw_query)
            if query.routing == RoutingStrategy.MULTI_STEP:
                futures["decompose"] = pool.submit(self._decompose_query, query.raw_query)
            for key, future in futures.items():
                try:
                    result = future.result(timeout=15)
                    if key == "multi_queries":
                        query.multi_queries = result
                    elif key == "hyde":
                        query.hyde_answer = result
                    elif key == "step_back":
                        query.step_back_query = result
                    elif key == "decompose":
                        query.sub_queries = result
                except Exception as e:
                    print(f"  ⚠ Query transform '{key}' failed: {e}")

    def _generate_multi_queries(self, query: str) -> List[str]:
        """Generate query variants for RAG-Fusion.
        Validates that generated queries are semantically relevant to the original.
        For broad 'tell me everything about X' queries, generates targeted sub-queries."""
        if self.llm.model is None:
            return [query]  # Return original if no LLM

        query_lower = query.lower().strip()

        # ── Entity-aware targeted sub-queries for broad queries ──────────
        # "Tell me everything about X" / "What do you know about X" patterns
        broad_patterns = [
            r"(?:tell me (?:everything|all) about|what (?:do you )?know about|"
            r"everything (?:about|on|regarding)|all about|summarize (?:everything about)?)"
            r"\s+(.+)",
        ]
        entity_name = None
        for pattern in broad_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entity_name = match.group(1).strip().rstrip("?.!")
                break

        # Also detect "Who is X" patterns
        who_match = re.search(r"who is\s+(.+)", query_lower)
        if who_match:
            entity_name = who_match.group(1).strip().rstrip("?.!")

        if entity_name:
            # Generate targeted sub-queries instead of relying on LLM
            entity_cap = entity_name.title()
            return [
                f"What are {entity_cap}'s projects and technical work?",
                f"What is {entity_cap}'s education and background?",
                f"What are {entity_cap}'s skills and interests?",
                f"What is {entity_cap}'s experience and achievements?",
            ]

        # ── Standard multi-query generation via LLM ─────────────────────
        prompt = PromptBuilder.multi_query_generation(query)

        result = self.llm.generate(prompt, max_tokens=150, temperature=0.3)
        lines = [l.strip() for l in result.split("\n") if l.strip()]

        variants = []
        # Extract key nouns/entities from original query for relevance check
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        # Remove common stop words
        stop_words = {
            "the", "and", "for", "are", "was", "were", "has", "have", "had",
            "been", "being", "what", "when", "where", "which", "who", "whom",
            "this", "that", "these", "those", "with", "from", "about", "into",
            "how", "does", "did", "can", "could", "would", "should", "will",
            "your", "you", "they", "them", "their", "some", "any", "all",
            "tell", "know", "think", "feel", "like", "more", "most", "very",
            "just", "also", "too", "than", "then", "now", "here", "there",
        }
        query_content_words = query_words - stop_words

        for line in lines:
            # Clean up numbering
            clean = re.sub(r'^(Version\s*)?\d+[:.]\s*', '', line).strip()
            if not clean or len(clean) < 5 or clean == query:
                continue

            # Relevance check: generated query must share at least 1 content word
            # with the original query (prevents completely off-topic generations)
            gen_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', clean.lower()))
            gen_content_words = gen_words - stop_words
            overlap = query_content_words & gen_content_words
            if overlap or not query_content_words:
                variants.append(clean)
            else:
                print(f"  ⚠ Discarding irrelevant multi-query: {clean[:60]}...")

        return variants[:3] if variants else [query]

    def _generate_hyde(self, query: str) -> str:
        """Generate hypothetical answer for HyDE."""
        if self.llm.model is None:
            return ""

        prompt = PromptBuilder.hyde_generation(query)
        return self.llm.generate(prompt, max_tokens=100, temperature=0.4).strip()

    def _generate_step_back(self, query: str) -> str:
        """Generate a step-back (more abstract) question."""
        if self.llm.model is None:
            return ""

        prompt = PromptBuilder.step_back_generation(query)
        return self.llm.generate(prompt, max_tokens=50, temperature=0.3).strip()

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        if self.llm.model is None:
            return [query]

        prompt = PromptBuilder.query_decomposition(query)

        result = self.llm.generate(prompt, max_tokens=150, temperature=0.3)
        lines = [l.strip() for l in result.split("\n") if l.strip()]

        sub_queries = []
        for line in lines:
            clean = re.sub(r'^\d+[:.]\s*', '', line).strip()
            if clean and len(clean) > 5:
                sub_queries.append(clean)

        return sub_queries[:3] if sub_queries else [query]
