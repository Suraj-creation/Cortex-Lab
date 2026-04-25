from __future__ import annotations

import json

import pytest

pytest.importorskip("networkx")

from src.storage.knowledge_graph import KnowledgeGraph


def test_knowledge_graph_loads_node_link_payload_with_edges_key(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "entity-1",
                "canonical_name": "Cortex Lab",
                "entity_type": "organization",
                "memory_ids": ["mem-1"],
            },
            {
                "id": "entity-2",
                "canonical_name": "Eva",
                "entity_type": "person",
                "memory_ids": ["mem-2"],
            },
        ],
        "edges": [
            {
                "source": "entity-1",
                "target": "entity-2",
                "relation": "created",
                "weight": 1.0,
            }
        ],
    }

    with open(graph_dir / "knowledge_graph.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    graph = KnowledgeGraph(data_dir=str(graph_dir))
    stats = graph.get_stats()
    exported = graph.get_graph_data()

    assert stats["nodes"] == 2
    assert stats["edges"] == 1
    assert {node["label"] for node in exported["nodes"]} == {"Cortex Lab", "Eva"}
    assert exported["edges"] == [
        {
            "source": "entity-1",
            "target": "entity-2",
            "relation": "created",
            "weight": 1.0,
        }
    ]
