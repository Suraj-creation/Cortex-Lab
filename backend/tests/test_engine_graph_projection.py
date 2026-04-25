from __future__ import annotations


class _EmptyKnowledgeGraph:
    def get_graph_data(self):
        return {"nodes": [], "edges": []}


class _MetadataBackedStore:
    def get_entities(self, limit: int = 100):
        assert limit >= 1000
        return [
            {
                "id": "entity-1",
                "canonical_name": "Cortex Lab",
                "entity_type": "organization",
                "memory_ids": ["mem-1", "mem-2"],
                "first_seen": "2026-04-24T00:00:00",
                "last_seen": "2026-04-25T00:00:00",
            }
        ]

    def get_edges(self, entity_id=None):
        assert entity_id is None
        return [
            {
                "source_id": "entity-1",
                "target_id": "entity-2",
                "relation": "created",
                "weight": 0.9,
            }
        ]


def test_engine_projects_graph_from_metadata_when_runtime_graph_is_empty():
    from src.engine import CortexRAGEngine

    engine = CortexRAGEngine.__new__(CortexRAGEngine)
    engine.initialized = True
    engine.knowledge_graph = _EmptyKnowledgeGraph()
    engine.metadata_store = _MetadataBackedStore()

    projected = engine.get_graph_data()

    assert projected["nodes"] == [
        {
            "id": "entity-1",
            "label": "Cortex Lab",
            "type": "organization",
            "memory_count": 2,
            "mentions": 2,
            "firstSeen": "2026-04-24T00:00:00",
            "lastSeen": "2026-04-25T00:00:00",
        },
        {
            "id": "entity-2",
            "label": "entity-2",
            "type": "unknown",
            "memory_count": 0,
            "mentions": 0,
            "firstSeen": None,
            "lastSeen": None,
        },
    ]
    assert projected["edges"] == [
        {
            "source": "entity-1",
            "target": "entity-2",
            "relation": "created",
            "weight": 0.9,
        }
    ]
