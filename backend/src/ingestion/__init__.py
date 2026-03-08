"""
Memory Ingestion Pipeline for Cortex Lab
Fine-tuned model integration for enriched memory processing:
- Memory type classification
- Emotion detection
- Entity extraction
- Topic extraction
- Proposition decomposition (atomic facts, EMNLP 2024)
- Contextual chunking (Anthropic-style)
- Semantic chunking (sentence-boundary aware)
- Embedding generation (BGE-large-en-v1.5, 1024d)
- Belief evolution detection (Stage 5 fine-tuning)
"""

import re
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.models import (
    CausalMemoryObject, MemoryType, EmotionLabel, EntityNode, GraphEdge
)
from src.models.embeddings import EmbeddingModel
from src.llm import LocalLLM
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph
from src.prompts import PromptBuilder, sanitize


class MemoryIngestionPipeline:
    """
    Full ingestion pipeline: raw text → enriched CausalMemoryObject → stored.
    
    Pipeline stages:
    1. Text preprocessing + semantic chunking
    2. Memory type classification
    3. Emotion detection
    4. Entity extraction
    5. Topic extraction
    6. Importance scoring
    7. Proposition decomposition (atomic facts)
    8. Contextual prefix generation
    9. Embedding generation (BGE-large passage embedding)
    10. Storage (vector + metadata + graph)
    11. Belief evolution detection (Stage 5 fine-tuning)
    """

    def __init__(self, llm: LocalLLM, embedding_model: EmbeddingModel,
                 vector_store: VectorStore, metadata_store: MetadataStore,
                 knowledge_graph: KnowledgeGraph):
        self.llm = llm
        self.embeddings = embedding_model
        self.vectors = vector_store
        self.metadata = metadata_store
        self.graph = knowledge_graph

        # Keyword-based classifiers (fast, no LLM needed for ~85% of cases)
        # NOTE: Avoid duplicate keywords across emotion categories to prevent conflicts
        self._emotion_keywords = {
            EmotionLabel.HAPPY: ["happy", "joy", "great", "wonderful", "love", "amazing",
                                 "glad", "pleased", "delighted", "enjoyed", "fun", "nice",
                                 "pleasant", "awesome", "fantastic", "blessed",
                                 "coffee with"],
            EmotionLabel.SAD: ["sad", "depressed", "down", "unhappy", "miserable", "grief",
                               "loss", "miss", "heartbroken", "sorrow", "devastated"],
            EmotionLabel.ANGRY: ["angry", "furious", "mad", "rage", "outraged", "infuriated",
                                 "livid", "hostile"],
            EmotionLabel.ANXIOUS: ["anxious", "worried", "nervous", "stress", "panic", "fear",
                                   "uncertain", "afraid", "dreading", "uneasy",
                                   "apprehensive"],
            EmotionLabel.EXCITED: ["excited", "thrilled", "eager", "pumped", "can't wait",
                                   "stoked", "new paper", "breakthrough", "discovered",
                                   "launched", "published", "announced"],
            EmotionLabel.CONFUSED: ["confused", "puzzled", "unsure", "don't understand",
                                    "lost", "bewildered", "perplexed", "baffled"],
            EmotionLabel.HOPEFUL: ["hopeful", "optimistic", "promising", "looking forward",
                                   "positive", "encouraged", "inspired"],
            EmotionLabel.FRUSTRATED: ["frustrated", "stuck", "annoyed", "struggling",
                                      "difficult", "irritated", "toxic", "unreasonable",
                                      "doesn't listen", "seriously considering leaving",
                                      "fed up", "avoiding"],
        }

        self._memory_type_keywords = {
            MemoryType.EPISODIC: ["went to", "met with", "visited", "talked to", "saw"],
            MemoryType.SEMANTIC: ["learned", "understood", "concept", "means", "defined as",
                                  "theory", "is that", "works by"],
            MemoryType.PROCEDURAL: ["how to", "process:", "steps", "method", "procedure",
                                    "workflow", "1)", "step 1"],
            MemoryType.REFLECTIVE: ["realized", "i think", "i feel", "i believe",
                                    "changed my mind", "pattern", "noticed",
                                    "i love", "i hate", "opinion", "perspective",
                                    "considering", "culture", "valued", "frustrated",
                                    "i'm frustrated", "doesn't matter",
                                    "seriously considering",
                                    "i prefer", "my philosophy", "what matters",
                                    "i've come to", "looking back"],
        }

    async def ingest(self, content: str, session_id: str = "",
                     source: str = "chat", session_context: str = "") -> CausalMemoryObject:
        """
        Full ingestion pipeline. Returns enriched memory object.
        For long content (>1000 chars), splits into semantic chunks.
        """
        t0 = time.time()

        # ── Content validation (§13.4) ───────────────────────────────────
        content = self._validate_content(content)
        if not content:
            return None

        # ── Semantic chunking for long documents ─────────────────────────
        if len(content) > 1000:
            chunks = self._chunk_long_content(content)
            if len(chunks) > 1:
                print(f"  📑 Semantic chunking: {len(content)} chars → {len(chunks)} chunks")
                first_memory = None
                for i, chunk in enumerate(chunks):
                    mem = await self._ingest_single(
                        chunk, session_id, source, session_context,
                        chunk_index=i, total_chunks=len(chunks)
                    )
                    if i == 0:
                        first_memory = mem
                elapsed_ms = (time.time() - t0) * 1000
                print(f"  ✓ Chunked ingestion complete ({elapsed_ms:.0f}ms)")
                return first_memory

        return await self._ingest_single(content, session_id, source, session_context)

    async def _ingest_single(self, content: str, session_id: str = "",
                              source: str = "chat", session_context: str = "",
                              chunk_index: int = 0, total_chunks: int = 1) -> CausalMemoryObject:
        """Ingest a single piece of content (or a chunk from a larger doc)."""
        t0 = time.time()

        # 1. Create base memory
        memory = CausalMemoryObject(
            content=content.strip(),
            session_id=session_id,
            source=source,
            timestamp=datetime.now(),
        )

        # 2. Classify memory type (keyword-first, LLM fallback)
        memory.memory_type = self._classify_memory_type(content)

        # 3. Detect emotion
        memory.emotion, memory.emotion_confidence = self._detect_emotion(content)

        # 4. Extract entities
        memory.entities = self._extract_entities(content)

        # 5. Extract topics
        memory.topics = self._extract_topics(content)

        # 6. Score importance
        memory.importance = self._score_importance(content, memory)

        # 7. Decompose into propositions
        memory.propositions = self._extract_propositions(content)

        # 8. Generate contextual prefix (Anthropic-style contextual enrichment)
        if session_context:
            memory.context_prefix = self._generate_context_prefix(content, session_context)
        elif len(content) > 60:
            memory.context_prefix = self._auto_context_prefix(content, memory)

        # Add chunk info to prefix if chunked
        if total_chunks > 1:
            chunk_label = f"[Chunk {chunk_index + 1}/{total_chunks}] "
            memory.context_prefix = chunk_label + (memory.context_prefix or "")

        # 9. Generate embedding (passage embedding for BGE asymmetric retrieval)
        embed_text = memory.context_prefix + " " + content if memory.context_prefix else content
        embedding = self.embeddings.embed_passage(embed_text)
        memory.embedding = embedding.tolist()

        # 10. Deduplication — skip if near-duplicate exists (>0.95 similarity)
        dedup_result = self._check_deduplication(embedding, memory)
        if dedup_result is not None:
            print(f"  ♻️  Dedup: skipped ingestion (sim={dedup_result:.3f}), updated existing memory")
            return None

        # 11. Store everything
        self.vectors.add(memory.id, embedding, memory.timestamp)
        self.metadata.store_memory(memory)

        # 12. Update knowledge graph with entities + typed causal edges
        self._update_graph(memory)

        elapsed_ms = (time.time() - t0) * 1000
        print(f"  📝 Memory ingested: [{memory.memory_type.value}] {content[:60]}... ({elapsed_ms:.0f}ms)")

        return memory

    def _chunk_long_content(self, content: str, max_chunk_chars: int = 800,
                             overlap_sentences: int = 1) -> List[str]:
        """Split long content into semantic chunks at sentence boundaries.
        Each chunk shares overlap_sentences with the next for continuity."""
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

            if current_len >= max_chunk_chars and i < len(sentences) - 1:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_len = sum(len(s) + 1 for s in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if len(chunks) > 1 else [content]

    async def consolidate_old_memories(self, max_age_days: int = 180,
                                        min_group_size: int = 3,
                                        max_groups: int = 20) -> int:
        """Consolidate old memories into topic-based summaries.
        Groups memories older than max_age_days by topic, generates
        LLM summaries, replaces originals with consolidated memories.
        Returns count of memories consolidated."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=max_age_days)
        all_memories = self.metadata.get_all_memories(limit=5000)

        old = [m for m in all_memories if m.timestamp < cutoff and m.raptor_level == 0]
        if len(old) < min_group_size:
            return 0

        groups: Dict[str, List] = {}
        for m in old:
            topic = m.topics[0] if m.topics else "general"
            groups.setdefault(topic, []).append(m)

        consolidated = 0
        for topic, memories in list(groups.items())[:max_groups]:
            if len(memories) < min_group_size:
                continue

            combined = "\n".join(m.content[:200] for m in memories[:15])
            if self.llm.model is not None or (hasattr(self.llm, 'has_gemini') and self.llm.has_gemini):
                summary = self.llm.summarize(combined, max_length=300)
            else:
                summary = " ".join(m.content.split(".")[0] + "." for m in memories[:5])

            if not summary or len(summary.strip()) < 20:
                continue

            consolidated_mem = CausalMemoryObject(
                content=f"[Consolidated: {topic}] {summary}",
                session_id="consolidation",
                source="consolidation",
                timestamp=memories[-1].timestamp,
                memory_type=MemoryType.SEMANTIC,
                topics=[topic],
                entities=list(set(e for m in memories for e in m.entities[:3]))[:10],
                importance=max(m.importance for m in memories),
            )

            embedding = self.embeddings.embed_passage(consolidated_mem.content)
            consolidated_mem.embedding = embedding.tolist()
            self.vectors.add(consolidated_mem.id, embedding, consolidated_mem.timestamp)
            self.metadata.store_memory(consolidated_mem)
            consolidated += len(memories)
            print(f"  🔄 Consolidated {len(memories)} memories → topic '{topic}'")

        return consolidated

    def _classify_memory_type(self, text: str) -> MemoryType:
        """Classify memory type using keyword matching."""
        text_lower = text.lower()
        scores = {}
        for mtype, keywords in self._memory_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[mtype] = score

        if scores:
            return max(scores, key=scores.get)

        # LLM fallback for ambiguous cases
        if self.llm.model is not None:
            prompt = PromptBuilder.classify_memory_type(text)
            result = self.llm.classify(
                prompt,
                ["episodic", "semantic", "procedural", "reflective"],
                default="episodic"
            )
            try:
                return MemoryType(result.lower().strip())
            except ValueError:
                pass

        return MemoryType.EPISODIC

    def _detect_emotion(self, text: str) -> Tuple[EmotionLabel, float]:
        """Detect emotion using keyword scoring."""
        text_lower = text.lower()
        scores = {}
        for emotion, keywords in self._emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score

        if scores:
            best = max(scores, key=scores.get)
            confidence = min(scores[best] / 3.0, 1.0)
            return best, confidence

        return EmotionLabel.NEUTRAL, 0.5

    # ── Tech Terms Dictionary for Entity Extraction (§Gap 3) ────────────
    # Lowercase tech terms, frameworks, tools, and acronyms that capitalization
    # heuristics miss. Organized by category for maintainability.
    _TECH_TERMS = {
        # Programming languages
        "python", "javascript", "typescript", "java", "c++", "c#", "golang",
        "rust", "ruby", "swift", "kotlin", "scala", "php", "perl", "lua",
        "haskell", "elixir", "clojure", "dart", "r",
        # Frameworks & libraries
        "react", "angular", "vue", "svelte", "nextjs", "next.js", "nuxt",
        "django", "flask", "fastapi", "express", "nestjs", "spring",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "langchain", "llamaindex", "huggingface", "transformers",
        "tailwind", "bootstrap", "jquery", "three.js",
        # Cloud & infrastructure
        "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "github", "gitlab", "bitbucket",
        "vercel", "netlify", "heroku", "railway", "supabase", "firebase",
        # Databases
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "duckdb", "sqlite", "cassandra", "dynamodb", "neo4j", "pinecone",
        "chromadb", "weaviate", "faiss", "milvus",
        # AI/ML concepts
        "ai", "ml", "nlp", "llm", "rag", "gpt", "bert", "gan", "cnn", "rnn",
        "lstm", "transformer", "diffusion", "reinforcement learning",
        "deep learning", "machine learning", "computer vision",
        "natural language processing", "neural network",
        # Tools & platforms
        "linux", "ubuntu", "macos", "windows", "nginx", "apache",
        "git", "npm", "pip", "conda", "yarn", "pnpm",
        "vscode", "vim", "neovim", "jupyter", "colab",
        "figma", "postman", "swagger", "grafana", "prometheus",
        # Protocols & standards
        "api", "rest", "graphql", "grpc", "websocket", "http", "tcp",
        "oauth", "jwt", "ssl", "tls", "ssh",
        # Hardware
        "gpu", "cpu", "tpu", "cuda", "vram", "nvidia", "amd", "intel",
        # Companies & services
        "google", "microsoft", "amazon", "meta", "apple", "openai",
        "anthropic", "deepseek", "mistral", "cohere", "hugging face",
    }

    # Multi-word tech terms that should be matched as phrases
    _TECH_PHRASES = {
        "deep learning", "machine learning", "computer vision",
        "natural language processing", "reinforcement learning",
        "neural network", "knowledge graph", "vector database",
        "large language model", "next.js", "three.js", "hugging face",
        "scikit-learn",
    }

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities using pattern matching, tech term dictionary,
        possessive stripping, and multi-word entity detection."""
        entities = []
        text_lower = text.lower()

        # ── Phase 1: Tech phrase matching (multi-word lowercase terms) ──
        for phrase in self._TECH_PHRASES:
            if phrase in text_lower:
                # Use title case for display
                entities.append(phrase.title() if len(phrase) > 3 else phrase.upper())

        # ── Phase 2: Tech term matching (single-word lowercase terms) ──
        words_lower = re.findall(r'\b[\w.#+]+\b', text_lower)
        for word in words_lower:
            if word in self._TECH_TERMS and word not in {p.split()[0] for p in self._TECH_PHRASES if p in text_lower}:
                # Skip if already captured as part of a tech phrase
                already_in = any(word in e.lower() for e in entities)
                if not already_in:
                    # Use original casing if available, else title/upper
                    if len(word) <= 4 and word.isalpha():
                        entities.append(word.upper())  # AI, ML, NLP, GPU etc.
                    else:
                        entities.append(word.title())   # Python, Docker, React etc.

        # ── Phase 3: Multi-word capitalized sequences (e.g. "Cortex Lab") ──
        multi_word = re.findall(r'(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        for mw in multi_word:
            clean_mw = re.sub(r"[''']s$", "", mw).strip()
            if len(clean_mw) > 2:
                already_in = any(clean_mw.lower() == e.lower() for e in entities)
                if not already_in:
                    entities.append(clean_mw)

        # ── Phase 4: Single capitalized words (proper nouns) ──
        words = text.split()
        for i, word in enumerate(words):
            word = re.sub(r"[''']s$", "", word)
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and i > 0 and len(clean) > 1:
                already_in = any(clean.lower() == e.lower() for e in entities)
                if not already_in and not any(clean in mw for mw in entities):
                    entities.append(clean)

        # ── Phase 5: Quoted strings ──
        quoted = re.findall(r'"([^"]+)"', text)
        for q in quoted:
            if q.lower() not in {e.lower() for e in entities}:
                entities.append(q)

        # Deduplicate (case-insensitive) and cap
        seen = set()
        unique = []
        for e in entities:
            key = e.lower().strip()
            if key not in seen and len(key) > 1:
                seen.add(key)
                unique.append(e)

        return unique[:15]  # Increased cap from 10 → 15 for richer entity coverage

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics using simple keyword/category matching."""
        topic_keywords = {
            "work": ["work", "job", "office", "meeting", "project", "deadline", "colleague", "boss", "career"],
            "health": ["health", "exercise", "gym", "doctor", "sleep", "diet", "sick", "medicine", "workout"],
            "relationships": ["friend", "family", "partner", "relationship", "love", "date", "social"],
            "learning": ["learn", "study", "course", "book", "read", "understand", "tutorial", "research"],
            "technology": ["code", "programming", "software", "computer", "AI", "machine learning", "model", "algorithm"],
            "finance": ["money", "budget", "invest", "salary", "expense", "save", "cost", "price"],
            "personal": ["feel", "think", "believe", "want", "goal", "dream", "plan", "decide"],
            "creative": ["write", "design", "art", "music", "create", "build", "idea", "project"],
        }

        text_lower = text.lower()
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics[:5]

    def _score_importance(self, text: str, memory: CausalMemoryObject) -> float:
        """Score importance 0.0-1.0 based on heuristics."""
        score = 0.5

        # Longer content tends to be more important
        word_count = len(text.split())
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.1

        # Emotional content is more important
        if memory.emotion != EmotionLabel.NEUTRAL:
            score += 0.1
        if memory.emotion_confidence > 0.7:
            score += 0.1

        # Reflective memories are higher importance
        if memory.memory_type == MemoryType.REFLECTIVE:
            score += 0.15

        # Entities increase importance
        if len(memory.entities) > 2:
            score += 0.1

        # Decision keywords
        decision_words = ["decided", "chose", "will", "plan to", "going to", "committed"]
        if any(w in text.lower() for w in decision_words):
            score += 0.15

        return min(max(score, 0.0), 1.0)

    def _extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic propositions using LLM-based decomposition (EMNLP 2024).
        Falls back to enhanced sentence splitting if LLM unavailable.
        Handles numbered lists (1) ... 2) ...) correctly.
        """
        # Try LLM-based atomic fact decomposition first
        if self.llm.model is not None and len(text) > 30:
            try:
                prompt = PromptBuilder.proposition_extraction(text)
                result = self.llm.generate(prompt, max_tokens=300, temperature=0.1)
                props = [p.strip().lstrip("- •·") for p in result.strip().split("\n")]
                props = [p for p in props if len(p) > 10 and not p.startswith("Atomic")]
                if props:
                    return props[:12]
            except Exception:
                pass

        # Enhanced fallback: split by clauses, sentences, AND numbered lists
        propositions = []

        # First: split numbered lists (1) ... 2) ... or 1. ... 2. ...)
        numbered_items = re.split(r'(?:^|\n)\s*\d+[).]\s+', text)
        if len(numbered_items) > 2:
            # It's a numbered list — each item is a proposition
            for item in numbered_items:
                item = item.strip().rstrip('.!?')
                if len(item) > 10:
                    propositions.append(item)
            return propositions[:12]

        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            # Further split compound sentences on conjunctions
            clauses = re.split(r'\s+(?:and|but|however|although|because|so|then)\s+', sent, flags=re.IGNORECASE)
            for clause in clauses:
                clause = clause.strip().rstrip('.!?')
                if len(clause) > 10:
                    propositions.append(clause)
        return propositions[:12]

    def _generate_context_prefix(self, content: str, session_context: str) -> str:
        """Generate contextual prefix using session context (Anthropic-style)."""
        if self.llm.model is not None and session_context:
            prompt = PromptBuilder.context_prefix(content, session_context)
            prefix = self.llm.generate(prompt, max_tokens=60, temperature=0.2)
            return prefix.strip()
        return ""

    def _auto_context_prefix(self, content: str, memory: 'CausalMemoryObject') -> str:
        """Auto-generate a minimal context prefix from memory metadata.
        Used when no session_context is available but we still want enrichment."""
        parts = []
        if memory.memory_type != MemoryType.EPISODIC:
            parts.append(f"This is a {memory.memory_type.value} memory.")
        if memory.topics:
            parts.append(f"Topics: {', '.join(memory.topics[:3])}.")
        if memory.entities:
            parts.append(f"Involves: {', '.join(memory.entities[:3])}.")
        return " ".join(parts) if parts else ""

    def _check_deduplication(self, embedding: np.ndarray, memory: 'CausalMemoryObject') -> Optional[float]:
        """Check if a near-duplicate memory exists (cosine sim > 0.95).
        If so, update the existing memory's timestamp and return the similarity score.
        Returns None if no duplicate found (memory should be stored)."""
        if self.vectors.count() == 0:
            return None

        similar = self.vectors.search(embedding, top_k=5)
        for mem_id, sim_score in similar:
            if sim_score > 0.95 and mem_id != memory.id:
                # Near-duplicate — update existing memory's metadata instead
                try:
                    existing = self.metadata.get_memory(mem_id)
                    if existing:
                        # Update last_seen timestamp on the existing memory
                        self.metadata.update_memory_timestamp(mem_id, datetime.now())
                except Exception:
                    pass
                return sim_score

        return None

    def _update_graph(self, memory: CausalMemoryObject):
        """Update knowledge graph with extracted entities and relationships.
        Also syncs entity/edge records to DuckDB metadata store for dashboard stats."""
        for entity_name in memory.entities:
            # Check if entity already exists
            existing_id = self.graph.find_entity_by_name(entity_name)
            if existing_id:
                # Update existing entity's last_seen and memory_ids
                if self.graph.graph is not None and existing_id in self.graph.graph:
                    # Use nx.set_node_attributes for safe mutation (AtlasView compat)
                    import networkx as nx
                    nx.set_node_attributes(self.graph.graph, {existing_id: {"last_seen": memory.timestamp.isoformat()}})
                    existing_mids = list(self.graph.graph.nodes[existing_id].get("memory_ids", []))
                    if memory.id not in existing_mids:
                        existing_mids.append(memory.id)
                        nx.set_node_attributes(self.graph.graph, {existing_id: {"memory_ids": existing_mids}})
                    # Sync updated entity to DuckDB
                    try:
                        node_data = self.graph.graph.nodes[existing_id]
                        entity = EntityNode(
                            id=existing_id,
                            canonical_name=node_data.get("canonical_name", entity_name),
                            entity_type=node_data.get("entity_type", "unknown"),
                            first_seen=datetime.fromisoformat(node_data["first_seen"]) if isinstance(node_data.get("first_seen"), str) else memory.timestamp,
                            last_seen=memory.timestamp,
                            memory_ids=existing_mids,
                        )
                        self.metadata.store_entity(entity)
                    except Exception:
                        pass
            else:
                # Create new entity
                entity = EntityNode(
                    canonical_name=entity_name,
                    entity_type=self._infer_entity_type(entity_name, memory.content),
                    first_seen=memory.timestamp,
                    last_seen=memory.timestamp,
                    memory_ids=[memory.id],
                )
                self.graph.add_entity(entity)
                # Sync new entity to DuckDB
                try:
                    self.metadata.store_entity(entity)
                except Exception:
                    pass

        # Create edges between co-occurring entities
        for i, ent1 in enumerate(memory.entities):
            for ent2 in memory.entities[i + 1:]:
                id1 = self.graph.find_entity_by_name(ent1)
                id2 = self.graph.find_entity_by_name(ent2)
                if id1 and id2:
                    edge = GraphEdge(
                        source_id=id1,
                        target_id=id2,
                        relation=self._infer_relation(ent1, ent2, memory.content),
                        weight=1.0,
                        memory_ids=[memory.id],
                        timestamp=memory.timestamp,
                    )
                    self.graph.add_edge(edge)
                    # Sync edge to DuckDB
                    try:
                        self.metadata.store_edge(edge)
                    except Exception:
                        pass

        # Detect and store belief evolution
        self._detect_belief_evolution(memory)

    def _infer_entity_type(self, entity_name: str, context: str) -> str:
        """Infer entity type from context clues."""
        context_lower = context.lower()
        name_lower = entity_name.lower()

        # Person indicators
        person_words = ["met", "talked to", "said", "told me", "friend", "colleague", "manager", "boss", "partner"]
        if any(w in context_lower for w in person_words):
            return "person"

        # Place indicators
        place_words = ["went to", "visited", "at the", "in the", "location", "city", "country", "office"]
        if any(w in context_lower for w in place_words):
            return "place"

        # Project indicators
        project_words = ["project", "codebase", "repo", "app", "system", "framework", "tool"]
        if any(w in context_lower for w in project_words):
            return "project"

        # Concept/topic indicators
        concept_words = ["concept", "theory", "idea", "principle", "method", "approach"]
        if any(w in context_lower for w in concept_words):
            return "concept"

        return "unknown"

    def _infer_relation(self, entity1: str, entity2: str, context: str) -> str:
        """Infer the relation between two co-occurring entities.
        Returns typed causal edges when causal language is detected."""
        context_lower = context.lower()
        # Strong causal indicators → typed causal edges
        caused_words = ["because of", "caused by", "was caused", "the reason"]
        if any(w in context_lower for w in caused_words):
            return "caused"
        led_to_words = ["led to", "resulted in", "which meant", "so that", "therefore"]
        if any(w in context_lower for w in led_to_words):
            return "led_to"
        influence_words = ["influenced", "affected", "impacted", "shaped"]
        if any(w in context_lower for w in influence_words):
            return "influenced"
        collab_words = ["worked with", "collaborated", "together", "team"]
        if any(w in context_lower for w in collab_words):
            return "works_with"
        discuss_words = ["discussed", "talked about", "mentioned", "about"]
        if any(w in context_lower for w in discuss_words):
            return "discussed"
        return "co_mentioned"

    def _detect_belief_evolution(self, memory: CausalMemoryObject):
        """
        Detect belief contradictions/evolution when a new memory is ingested.
        Uses fine-tuned detect_belief_change (Stage 5) when available.
        
        Multi-stage pipeline per RAG-Architecture §9:
        1. Semantic similarity: find memories about same topic (>0.75)
        2. Stance detection: classify stance change (keyword + LLM Stage 5)
        3. Temporal context: weight recency, require minimum gap
        4. Classification: CONTRADICTION / REFINEMENT / EXPANSION / ABANDONMENT
        5. Storage as BeliefDelta
        """
        from src.models import BeliefDelta, BeliefChangeType

        if not memory.embedding or memory.memory_type not in (MemoryType.REFLECTIVE, MemoryType.SEMANTIC):
            return

        # Stage 1: Find semantically similar past memories (same topic)
        query_emb = self.embeddings.embed(memory.content)
        similar = self.vectors.search(query_emb, top_k=10)

        for mem_id, sim_score in similar:
            if sim_score < 0.75 or mem_id == memory.id:
                continue

            old_memory = self.metadata.get_memory(mem_id)
            if not old_memory:
                continue

            # Skip if same session (likely continuation, not contradiction)
            if old_memory.session_id == memory.session_id and memory.session_id:
                continue

            # Stage 3: Temporal context — require minimum time gap (1 day)
            time_gap = abs((memory.timestamp - old_memory.timestamp).total_seconds())
            if time_gap < 86400:
                continue

            # Stage 2 + 4: Use fine-tuned LLM belief change detection (Stage 5) if available
            topic = memory.topics[0] if memory.topics else "general"

            if self.llm.model is not None:
                try:
                    delta_result = self.llm.detect_belief_change(
                        old_memory.content[:300], memory.content[:300], topic
                    )
                    change_type_str = delta_result.get("change_type", "none").lower()
                    explanation = delta_result.get("explanation", "")
                    llm_confidence = delta_result.get("confidence", 0.5)

                    type_map = {
                        "contradiction": BeliefChangeType.CONTRADICTION,
                        "refinement": BeliefChangeType.REFINEMENT,
                        "reinforcement": BeliefChangeType.REINFORCEMENT,
                        "new_belief": BeliefChangeType.NEW_BELIEF,
                    }
                    if change_type_str in type_map:
                        change_type = type_map[change_type_str]
                        confidence = llm_confidence
                    elif change_type_str in ("none", "no_change"):
                        continue
                    else:
                        # Fall back to keyword stance
                        stance = self._detect_stance(old_memory.content, memory.content)
                        if stance == "disagree":
                            change_type = BeliefChangeType.CONTRADICTION
                            confidence = min(sim_score + 0.1, 1.0)
                        elif stance == "expand":
                            change_type = BeliefChangeType.REFINEMENT
                            confidence = sim_score * 0.8
                        else:
                            continue
                except Exception:
                    # LLM failed, fall back to keyword detection
                    stance = self._detect_stance(old_memory.content, memory.content)
                    if stance == "disagree":
                        change_type = BeliefChangeType.CONTRADICTION
                        confidence = min(sim_score + 0.1, 1.0)
                    elif stance == "expand":
                        change_type = BeliefChangeType.REFINEMENT
                        confidence = sim_score * 0.8
                    else:
                        continue
                    explanation = ""
            else:
                # No LLM: keyword-based stance detection
                stance = self._detect_stance(old_memory.content, memory.content)
                if stance == "disagree":
                    change_type = BeliefChangeType.CONTRADICTION
                    confidence = min(sim_score + 0.1, 1.0)
                elif stance == "expand":
                    change_type = BeliefChangeType.REFINEMENT
                    confidence = sim_score * 0.8
                else:
                    continue
                explanation = ""

            # Stage 5: Store BeliefDelta
            delta = BeliefDelta(
                topic=topic,
                old_belief_id=old_memory.id,
                new_belief_id=memory.id,
                old_belief_text=old_memory.content[:200],
                new_belief_text=memory.content[:200],
                change_type=change_type,
                confidence=confidence,
                detected_at=datetime.now(),
                evidence_chain=[old_memory.id, memory.id],
            )
            self.metadata.store_belief_delta(delta)
            print(f"  🔄 Belief evolution detected: {change_type.value} on '{topic}' (conf: {confidence:.2f})")
            break  # Only detect the most significant change per ingestion

    def _detect_stance(self, old_text: str, new_text: str) -> str:
        """Detect stance between two texts: agree, disagree, expand, neutral."""
        old_lower = old_text.lower()
        new_lower = new_text.lower()

        # Strong disagreement indicators
        negation_pairs = [
            ("love", "hate"), ("like", "dislike"), ("good", "bad"), ("great", "terrible"),
            ("agree", "disagree"), ("support", "oppose"), ("yes", "no"),
            ("happy", "unhappy"), ("positive", "negative"), ("enjoy", "dread"),
        ]
        for pos, neg in negation_pairs:
            if (pos in old_lower and neg in new_lower) or (neg in old_lower and pos in new_lower):
                return "disagree"

        # Explicit contradiction words in new text
        contradiction_words = ["actually", "i was wrong", "changed my mind", "no longer",
                                "not anymore", "contrary to", "opposite", "however"]
        if any(w in new_lower for w in contradiction_words):
            return "disagree"

        # Expansion indicators
        expand_words = ["also", "additionally", "moreover", "learned that", "realized",
                        "in addition", "furthermore", "building on"]
        if any(w in new_lower for w in expand_words):
            return "expand"

        return "neutral"

    # ── Content Validation (§13.4) ───────────────────────────────────────

    _MAX_MEMORY_LENGTH = 10000  # 10K chars

    def _validate_content(self, content: str) -> Optional[str]:
        """Validate and sanitize memory content before ingestion.
        Returns cleaned content or None if invalid."""
        if not content or not isinstance(content, str):
            return None

        content = content.strip()

        # Reject empty or trivially short content
        if len(content) < 2:
            return None

        # Truncate excessively long content
        if len(content) > self._MAX_MEMORY_LENGTH:
            content = content[:self._MAX_MEMORY_LENGTH] + "... [truncated]"
            print(f"  ⚠ Memory content truncated to {self._MAX_MEMORY_LENGTH} chars")

        # Remove null bytes and non-printable characters (except newlines/tabs)
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

        # Strip prompt template markers to prevent injection (§14.1)
        for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"]:
            content = content.replace(marker, "")

        return content.strip() if content.strip() else None

    # ── RAPTOR Tree Indexing ─────────────────────────────────────────────

    def build_raptor_clusters(self, min_cluster_size: int = 5, max_clusters: int = 20):
        """
        RAPTOR hierarchical indexing: cluster leaf memories by topic similarity
        using K-means on embeddings, generate a summary for each cluster,
        and store the summary as a new memory with raptor_level=1.

        Called by engine.py when memory count crosses thresholds (50, 200, 1000).
        """
        from sklearn.cluster import KMeans

        # Get all leaf memories (raptor_level=0) with embeddings
        all_memories = self.metadata.get_all_memories(limit=5000)
        leaf_memories = [m for m in all_memories if m.raptor_level == 0]

        if len(leaf_memories) < min_cluster_size * 2:
            print(f"  ℹ RAPTOR: not enough leaf memories ({len(leaf_memories)}), skipping")
            return

        # Collect embeddings
        mem_ids = []
        embeddings = []
        for mem in leaf_memories:
            if mem.id in self.vectors.vectors:
                mem_ids.append(mem.id)
                embeddings.append(self.vectors.vectors[mem.id])

        if len(embeddings) < min_cluster_size * 2:
            return

        X = np.array(embeddings, dtype=np.float32)

        # Determine number of clusters
        n_clusters = min(max_clusters, max(2, len(embeddings) // min_cluster_size))

        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
        except Exception as e:
            print(f"  ⚠ RAPTOR clustering failed: {e}")
            return

        # Build clusters
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        created = 0
        for cluster_id, member_indices in clusters.items():
            if len(member_indices) < min_cluster_size:
                continue

            # Collect child memory texts for summarization
            child_ids = [mem_ids[i] for i in member_indices]
            child_texts = []
            for mid in child_ids[:10]:  # Cap at 10 for LLM context
                mem = self.metadata.get_memory(mid)
                if mem:
                    child_texts.append(mem.content[:200])

            if not child_texts:
                continue

            # Generate summary via LLM
            combined = "\n".join(f"- {t}" for t in child_texts)
            if self.llm.model is not None:
                prompt = PromptBuilder.raptor_summary(combined)
                # Use higher token budget so the model can finish <think> + produce summary
                summary_text = self.llm.generate(prompt, max_tokens=400, temperature=0.2)
            else:
                summary_text = f"Cluster of {len(child_ids)} related memories about: {', '.join(child_texts[0].split()[:10])}"

            # ── Clean LLM output: strip <think> blocks, artifacts ──
            if summary_text:
                import re as _re
                # Strip <think>...</think> blocks (DeepSeek-R1 reasoning traces)
                summary_text = _re.sub(
                    r'<think>.*?</think>', '', summary_text, flags=_re.DOTALL
                ).strip()
                # If model only produced <think> without closing tag, discard entirely
                if '<think>' in summary_text:
                    summary_text = summary_text[:summary_text.index('<think>')].strip()
                # Remove other model artifacts
                for token in ['<|im_end|>', '<|im_start|>', '<|endoftext|>',
                              '<|im_sep|>', '<｜end▁of▁sentence｜>']:
                    summary_text = summary_text.replace(token, '')
                summary_text = summary_text.strip()

            if not summary_text or len(summary_text.strip()) < 20:
                continue

            # Create RAPTOR summary memory
            summary_memory = CausalMemoryObject(
                content=summary_text.strip(),
                memory_type=MemoryType.SEMANTIC,
                raptor_level=1,
                raptor_children=child_ids,
                source="raptor",
                timestamp=datetime.now(),
                topics=self._extract_topics(summary_text),
                importance=0.8,
            )

            # Embed and store the summary
            summary_embedding = self.embeddings.embed_passage(summary_text)
            summary_memory.embedding = summary_embedding.tolist()
            self.vectors.add(summary_memory.id, summary_embedding, summary_memory.timestamp)
            self.metadata.store_memory(summary_memory)
            created += 1

        if created > 0:
            print(f"  🌳 RAPTOR: created {created} cluster summaries from {len(leaf_memories)} memories")
