"""
Knowledge Graph for Cortex Lab
NetworkX-based entity-relationship graph with GraphRAG capabilities.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from src.models import EntityNode, GraphEdge


class KnowledgeGraph:
    """
    Entity-relationship knowledge graph using NetworkX.
    Supports:
    - Entity nodes with attributes
    - Typed edges (works_with, caused, discussed, etc.)
    - Multi-hop traversal for causal reasoning
    - Community detection for topic clustering
    - Graph serialization/deserialization
    """

    def __init__(self, data_dir: str = "data/graph"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Inverted name/alias index for O(1) entity lookups (§3.5)
        self._name_index: Dict[str, str] = {}   # lowercase_name -> entity_id
        self._alias_index: Dict[str, str] = {}  # lowercase_alias -> entity_id

        # Auto-save: save after every N mutations to prevent data loss on crash
        self._mutation_count = 0
        self._auto_save_interval = 20  # Save every 20 mutations

        if HAS_NX:
            self.graph = nx.DiGraph()
            print("  ✓ Knowledge graph initialized (NetworkX)")
        else:
            self.graph = None
            self._nodes: Dict[str, Dict] = {}
            self._edges: List[Dict] = []
            print("  ⚠ NetworkX not available, using basic graph fallback")

        self._load()
        self._rebuild_name_index()

    def add_entity(self, entity: EntityNode):
        """Add or update an entity node."""
        if self.graph is not None:
            self.graph.add_node(
                entity.id,
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                entity_type=entity.entity_type,
                first_seen=entity.first_seen.isoformat(),
                last_seen=entity.last_seen.isoformat(),
                memory_ids=entity.memory_ids,
                attributes=entity.attributes,
            )
        else:
            self._nodes[entity.id] = {
                "canonical_name": entity.canonical_name,
                "aliases": entity.aliases,
                "entity_type": entity.entity_type,
                "memory_ids": entity.memory_ids,
            }
        # Update name/alias index (§3.5)
        self._name_index[entity.canonical_name.lower()] = entity.id
        for alias in entity.aliases:
            self._alias_index[alias.lower()] = entity.id
        self._maybe_auto_save()

    def add_edge(self, edge: GraphEdge):
        """Add or update a relationship edge."""
        if self.graph is not None:
            # Merge with existing edge if present
            if self.graph.has_edge(edge.source_id, edge.target_id):
                edge_data = self.graph.edges[edge.source_id, edge.target_id]
                existing_mids = edge_data.get("memory_ids", [])
                merged_mids = list(set(existing_mids + edge.memory_ids))
                edge_data["memory_ids"] = merged_mids
                edge_data["weight"] = edge_data.get("weight", 1.0) + edge.weight
            else:
                self.graph.add_edge(
                    edge.source_id, edge.target_id,
                    relation=edge.relation,
                    weight=edge.weight,
                    memory_ids=edge.memory_ids,
                    timestamp=edge.timestamp.isoformat(),
                )
        else:
            self._edges.append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "weight": edge.weight,
                "memory_ids": edge.memory_ids,
            })
        self._maybe_auto_save()

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> List[Dict]:
        """Get all entities within max_hops of the given entity."""
        if self.graph is None or entity_id not in self.graph:
            return []

        visited: Set[str] = set()
        results = []
        queue = [(entity_id, 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)

            if current != entity_id:
                node_data = dict(self.graph.nodes[current])
                node_data["id"] = current
                node_data["hop_distance"] = depth
                results.append(node_data)

            # Traverse both directions
            for neighbor in list(self.graph.successors(current)) + list(self.graph.predecessors(current)):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return results

    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two entities."""
        if self.graph is None:
            return []
        try:
            return list(nx.shortest_path(self.graph, source_id, target_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_causal_chain(self, entity_id: str, direction: str = "backward") -> List[Dict]:
        """Trace causal chains from an entity."""
        if self.graph is None:
            return []

        chain = []
        visited = set()
        queue = [entity_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if direction == "backward":
                predecessors = list(self.graph.predecessors(current))
            else:
                predecessors = list(self.graph.successors(current))

            for pred in predecessors:
                edge_data = self.graph[pred][current] if direction == "backward" else self.graph[current][pred]
                if edge_data.get("relation") in ("caused", "led_to", "influenced", "resulted_in"):
                    node_data = dict(self.graph.nodes.get(pred, {}))
                    node_data["id"] = pred
                    node_data["relation"] = edge_data.get("relation", "related")
                    chain.append(node_data)
                    queue.append(pred)

        return chain

    def get_entity_memories(self, entity_id: str) -> List[str]:
        """Get all memory IDs associated with an entity."""
        if self.graph is not None and entity_id in self.graph:
            return self.graph.nodes[entity_id].get("memory_ids", [])
        return []

    def find_entity_by_name(self, name: str) -> Optional[str]:
        """Find entity ID by canonical name, alias, or fuzzy match.
        Uses O(1) inverted index for exact/alias matches (§3.5)."""
        name_lower = name.lower()

        # Stage 1: O(1) exact match on canonical name via index
        if name_lower in self._name_index:
            return self._name_index[name_lower]

        # Stage 2: O(1) exact match on aliases via index
        if name_lower in self._alias_index:
            return self._alias_index[name_lower]

        # Stage 3: Fuzzy matching (only when index misses — rare)
        if self.graph is not None:
            best_match = None
            best_score = 0.0
            for node_id, data in self.graph.nodes(data=True):
                canonical = data.get("canonical_name", "").lower()
                # Substring match
                if name_lower in canonical or canonical in name_lower:
                    score = len(min(name_lower, canonical, key=len)) / max(len(name_lower), len(canonical), 1)
                    if score > 0.6 and score > best_score:
                        best_score = score
                        best_match = node_id
                # Character-level similarity (simplified Jaro-like)
                else:
                    common = sum(1 for c in name_lower if c in canonical)
                    score = (2.0 * common) / (len(name_lower) + len(canonical) + 1e-10)
                    if score > 0.75 and score > best_score:
                        best_score = score
                        best_match = node_id

            if best_match:
                # Auto-add as alias for faster future lookups
                aliases = list(self.graph.nodes[best_match].get("aliases", []))
                if name not in aliases:
                    aliases.append(name)
                    nx.set_node_attributes(self.graph, {best_match: {"aliases": aliases}})
                # Update alias index
                self._alias_index[name_lower] = best_match
                return best_match

        return None

    def _rebuild_name_index(self):
        """Rebuild the inverted name/alias index from graph nodes (§3.5)."""
        self._name_index.clear()
        self._alias_index.clear()
        if self.graph is not None:
            for nid, data in self.graph.nodes(data=True):
                name = data.get("canonical_name", "").lower()
                if name:
                    self._name_index[name] = nid
                for alias in data.get("aliases", []):
                    if alias:
                        self._alias_index[alias.lower()] = nid

    def merge_entities(self, keep_id: str, merge_id: str):
        """Merge two entity nodes (merge_id → keep_id)."""
        if self.graph is None or keep_id not in self.graph or merge_id not in self.graph:
            return

        keep_data = dict(self.graph.nodes[keep_id])
        merge_data = dict(self.graph.nodes[merge_id])

        # Merge aliases
        existing_aliases = keep_data.get("aliases", [])
        merge_aliases = merge_data.get("aliases", [])
        merge_name = merge_data.get("canonical_name", "")
        all_aliases = list(set(existing_aliases + merge_aliases + ([merge_name] if merge_name else [])))
        nx.set_node_attributes(self.graph, {keep_id: {"aliases": all_aliases}})

        # Merge memory_ids
        keep_mids = set(keep_data.get("memory_ids", []))
        merge_mids = merge_data.get("memory_ids", [])
        nx.set_node_attributes(self.graph, {keep_id: {"memory_ids": list(keep_mids | set(merge_mids))}})

        # Redirect edges
        for pred in list(self.graph.predecessors(merge_id)):
            edge_data = dict(self.graph[pred][merge_id])
            if not self.graph.has_edge(pred, keep_id):
                self.graph.add_edge(pred, keep_id, **edge_data)
        for succ in list(self.graph.successors(merge_id)):
            edge_data = dict(self.graph[merge_id][succ])
            if not self.graph.has_edge(keep_id, succ):
                self.graph.add_edge(keep_id, succ, **edge_data)

        # Remove merged node
        self.graph.remove_node(merge_id)

    def get_community_summaries(self) -> List[Dict]:
        """Get community clusters with entity names (GraphRAG-style)."""
        communities = self.get_communities()
        summaries = []
        for i, community in enumerate(communities):
            members = []
            for node_id in community:
                if node_id in self.graph:
                    members.append(self.graph.nodes[node_id].get("canonical_name", node_id[:8]))
            all_mids = set()
            for node_id in community:
                if node_id in self.graph:
                    all_mids.update(self.graph.nodes[node_id].get("memory_ids", []))
            summaries.append({
                "community_id": i,
                "members": members,
                "size": len(community),
                "memory_count": len(all_mids),
                "memory_ids": list(all_mids)[:20],
            })
        return summaries

    def get_communities(self) -> List[List[str]]:
        """Detect communities using greedy modularity."""
        if self.graph is None or len(self.graph) < 2:
            return []
        try:
            undirected = self.graph.to_undirected()
            communities = list(nx.community.greedy_modularity_communities(undirected))
            return [list(c) for c in communities]
        except Exception:
            return []

    def get_graph_data(self) -> Dict:
        """Export full graph for visualization."""
        if self.graph is not None:
            nodes = []
            for nid, data in self.graph.nodes(data=True):
                nodes.append({
                    "id": nid,
                    "label": data.get("canonical_name", nid[:8]),
                    "type": data.get("entity_type", "unknown"),
                    "memory_count": len(data.get("memory_ids", [])),
                })
            edges = []
            for src, tgt, data in self.graph.edges(data=True):
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relation": data.get("relation", "related"),
                    "weight": data.get("weight", 1.0),
                })
            return {"nodes": nodes, "edges": edges}
        return {"nodes": [], "edges": []}

    def get_stats(self) -> Dict:
        if self.graph is not None:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": round(nx.density(self.graph), 4) if self.graph.number_of_nodes() > 0 else 0,
            }
        return {"nodes": len(self._nodes), "edges": len(self._edges)}

    def _maybe_auto_save(self):
        """Auto-save after every N mutations to prevent data loss on crash."""
        self._mutation_count += 1
        if self._mutation_count >= self._auto_save_interval:
            self._mutation_count = 0
            try:
                self.save()
            except Exception as e:
                print(f"  ⚠ KG auto-save failed: {e}")

    def save(self):
        if self.graph is not None:
            data = nx.node_link_data(self.graph)
            data["multigraph"] = False  # Always save as simple DiGraph
            with open(os.path.join(self.data_dir, "knowledge_graph.json"), "w") as f:
                json.dump(data, f, default=str)
            print(f"  ✓ Knowledge graph saved ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

    def _load(self):
        path = os.path.join(self.data_dir, "knowledge_graph.json")
        if os.path.exists(path) and self.graph is not None:
            try:
                with open(path) as f:
                    data = json.load(f)
                # Force multigraph=False so we always get a plain DiGraph
                data["multigraph"] = False
                loaded = nx.node_link_graph(data, directed=True)
                # Ensure we always have a DiGraph, not a MultiDiGraph
                if isinstance(loaded, nx.MultiDiGraph):
                    self.graph = nx.DiGraph(loaded)
                else:
                    self.graph = loaded
                print(f"  ✓ Knowledge graph loaded ({self.graph.number_of_nodes()} nodes)")
            except Exception as e:
                print(f"  ⚠ Failed to load graph: {e}")
