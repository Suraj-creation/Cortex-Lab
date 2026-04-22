"""Rebuild the knowledge graph from existing memories in DuckDB."""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph
from src.models import EntityNode, GraphEdge
import networkx as nx

data_dir = "data"
ms = MetadataStore(db_path=f"{data_dir}/cortex.duckdb")
graph = KnowledgeGraph(data_dir=f"{data_dir}/graph")

total = ms.count_memories()
print(f"Total memories: {total}")
print(f"Current graph: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges")


def extract_entities(text):
    entities = []
    words = text.split()
    for i, word in enumerate(words):
        word = re.sub(r"['']s$", "", word)
        clean = re.sub(r"[^\w]", "", word)
        if clean and clean[0].isupper() and i > 0 and len(clean) > 1:
            entities.append(clean)
    quoted = re.findall(r'"([^"]+)"', text)
    entities.extend(quoted)
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)
    return unique[:10]


def infer_entity_type(name, ctx):
    ctx_l = ctx.lower()
    for w in ["met", "friend", "colleague", "partner", "told me"]:
        if w in ctx_l:
            return "person"
    for w in ["visit", "went to", "city", "place", "location"]:
        if w in ctx_l:
            return "place"
    for w in ["project", "built", "created", "app", "tool", "platform"]:
        if w in ctx_l:
            return "project"
    for w in ["company", "org", "university", "school"]:
        if w in ctx_l:
            return "organization"
    for w in ["python", "react", "pytorch", "tensorflow", "javascript"]:
        if w in ctx_l:
            return "technology"
    return "concept"


def infer_relation(e1, e2, ctx):
    ctx_l = ctx.lower()
    for w in ["built", "created", "developed", "made"]:
        if w in ctx_l:
            return "created"
    for w in ["uses", "with", "using"]:
        if w in ctx_l:
            return "uses"
    return "related_to"


batch_size = 100
processed = 0
for offset in range(0, total, batch_size):
    memories = ms.get_all_memories(limit=batch_size, offset=offset)
    for mem in memories:
        entities = extract_entities(mem.content)
        for ent_name in entities:
            existing_id = graph.find_entity_by_name(ent_name)
            if existing_id:
                nx.set_node_attributes(
                    graph.graph,
                    {existing_id: {"last_seen": mem.timestamp.isoformat()}},
                )
                existing_mids = list(
                    graph.graph.nodes[existing_id].get("memory_ids", [])
                )
                if mem.id not in existing_mids:
                    existing_mids.append(mem.id)
                    nx.set_node_attributes(
                        graph.graph,
                        {existing_id: {"memory_ids": existing_mids}},
                    )
            else:
                entity = EntityNode(
                    canonical_name=ent_name,
                    entity_type=infer_entity_type(ent_name, mem.content),
                    first_seen=mem.timestamp,
                    last_seen=mem.timestamp,
                    memory_ids=[mem.id],
                )
                graph.add_entity(entity)

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                id1 = graph.find_entity_by_name(e1)
                id2 = graph.find_entity_by_name(e2)
                if id1 and id2:
                    edge = GraphEdge(
                        source_id=id1,
                        target_id=id2,
                        relation=infer_relation(e1, e2, mem.content),
                        weight=1.0,
                        memory_ids=[mem.id],
                        timestamp=mem.timestamp,
                    )
                    graph.add_edge(edge)
        processed += 1
    print(f"  Processed {processed}/{total} memories...")

print(
    f"\nGraph rebuilt: {graph.graph.number_of_nodes()} nodes, "
    f"{graph.graph.number_of_edges()} edges"
)
graph.save()
print("Graph saved to disk.")
