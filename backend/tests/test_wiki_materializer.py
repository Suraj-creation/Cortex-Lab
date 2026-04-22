"""
Tests for deterministic wiki materialization from ingested memories.

Run: python -m pytest tests/test_wiki_materializer.py -q
"""

from __future__ import annotations

from pathlib import Path


def _reset_singletons() -> None:
    from src.wiki.claim_store import ClaimStore
    from src.wiki.wiki_store import WikiStore

    ClaimStore._instance = None
    WikiStore._instance = None


def test_extract_claim_candidates_prefers_propositions() -> None:
    from src.wiki.materializer import extract_claim_candidates

    claims = extract_claim_candidates(
        content="This should not be used because propositions are provided.",
        propositions=[
            "The backend uses FastAPI for API routes.",
            "The mobile app uses React Native.",
        ],
        max_claims=8,
    )

    assert len(claims) == 2
    assert claims[0] == "The backend uses FastAPI for API routes."
    assert claims[1] == "The mobile app uses React Native."


def test_materialize_memory_into_wiki_creates_page_and_links_claims(tmp_path: Path) -> None:
    from src.wiki.claim_store import ClaimStore
    from src.wiki.materializer import materialize_memory_into_wiki
    from src.wiki.wiki_store import WikiStore

    _reset_singletons()

    wiki_dir = tmp_path / "wiki"
    claims_dir = wiki_dir / "claims"

    summary = materialize_memory_into_wiki(
        {
            "id": "mem-001",
            "content": "I am building a FastAPI backend and a React Native mobile app.",
            "source": "chat",
            "session_id": "session-abc",
            "topics": ["engineering"],
            "entities": ["FastAPI", "React Native"],
            "propositions": [
                "The backend is implemented with FastAPI.",
                "The mobile application is implemented with React Native.",
            ],
        },
        wiki_data_dir=str(wiki_dir),
        claims_data_dir=str(claims_dir),
    )

    assert summary["memory_id"] == "mem-001"
    assert summary["claims_extracted"] >= 2
    assert summary["claims_linked"] >= 2

    wiki_store = WikiStore.get_instance(data_dir=str(wiki_dir))
    pages = wiki_store.list_pages()
    assert len(pages) == 1
    page = wiki_store.get_page(pages[0]["id"])
    assert page is not None
    assert "## Stable Facts" in page.content
    assert len(page.claim_ids) >= 2

    claim_store = ClaimStore.get_instance(data_dir=str(claims_dir))
    stats = claim_store.stats()
    assert stats["total"] >= 2


def test_claim_store_flags_simple_negation_contradictions(tmp_path: Path) -> None:
    from src.wiki.claim_store import ClaimStore

    claims_dir = tmp_path / "claims"
    store = ClaimStore(data_dir=str(claims_dir))

    c1 = store.upsert(
        claim="I enjoy coffee in the mornings.",
        confidence=0.85,
        source_ids=["m1"],
        topic="habits",
    )
    c2 = store.upsert(
        claim="I do not enjoy coffee in the mornings.",
        confidence=0.85,
        source_ids=["m2"],
        topic="habits",
    )

    claim_1 = store.get_claim(c1)
    claim_2 = store.get_claim(c2)
    assert claim_1 is not None and claim_2 is not None
    assert c2 in claim_1.contradiction_ids
    assert c1 in claim_2.contradiction_ids
