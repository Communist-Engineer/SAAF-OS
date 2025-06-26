
# modules/contradiction/engine.py

"""
Contradiction Engine for SAAF-OS.

Detects, models, and resolves dialectical contradictions in goals, plans,
and world-state to guide synthesis rather than binary conflict resolution.
"""

import networkx as nx
from typing import Dict, Any, List

# Assuming these types are defined elsewhere in the project
GoalSet = Dict[str, Any]
WorldState = Dict[str, Any]
SynthesisPlan = Dict[str, Any]
FeedbackLog = Dict[str, Any]
ResolutionScore = float
ContradictionGraph = nx.Graph

class ContradictionEngine:
    """
    Implements the Contradiction Engine as specified in the documentation.
    """
    def __init__(self, ewma_alpha=0.3, 
                 labor_time_weight=0.4,
                 energy_efficiency_weight=0.3,
                 class_alignment_weight=0.3):
        self.graph = ContradictionGraph()
        self.tension_history = {}
        self.ewma_alpha = ewma_alpha
        self.labor_time_weight = labor_time_weight
        self.energy_efficiency_weight = energy_efficiency_weight
        self.class_alignment_weight = class_alignment_weight

    def detect_contradictions(self, nodes: Dict[str, Any], world_state: WorldState) -> ContradictionGraph:
        """
        Builds or updates the Contradiction Graph based on current nodes (goals, values, etc.) and world state.

        This implementation identifies contradictions based on nodes that affect the same entity
        in potentially conflicting ways (e.g., 'increase' vs. 'decrease' the same metric).
        """
        # Group nodes by the entities they affect
        affected_entities = {}
        for node_id, node_data in nodes.items():
            entity = node_data.get("entity")
            if entity:
                if entity not in affected_entities:
                    affected_entities[entity] = []
                affected_entities[entity].append(node_id)

        # Identify and add contradictions to the graph
        for entity, node_ids in affected_entities.items():
            if len(node_ids) > 1:
                # Add nodes to the graph if they don't exist
                for node_id in node_ids:
                    if not self.graph.has_node(node_id):
                        self.graph.add_node(node_id, **nodes[node_id])

                # Compare all pairs of nodes affecting the same entity
                import itertools
                for n1_id, n2_id in itertools.combinations(node_ids, 2):
                    n1 = nodes[n1_id]
                    n2 = nodes[n2_id]

                    # Example of a simple contradiction logic
                    # A real implementation would be more sophisticated
                    if n1.get("action") == "increase" and n2.get("action") == "decrease":
                        contradiction_type = "antagonistic"
                        tension = 0.8 # High tension for direct opposition
                    elif n1.get("action") == n2.get("action"):
                        contradiction_type = "synergistic"
                        tension = -0.2 # Negative tension for synergy
                    else:
                        contradiction_type = "related"
                        tension = 0.3 # Mild tension for relatedness

                    if not self.graph.has_edge(n1_id, n2_id):
                        self.graph.add_edge(
                            n1_id,
                            n2_id,
                            type=contradiction_type,
                            tension=tension,
                            history=[],
                            feasibility_cost=0.1 # Default feasibility cost
                        )

        return self.graph

    def resolve_contradiction(self, source_node: Any, target_node: Any, strategy: str = "synthesis") -> SynthesisPlan:
        """
        Proposes a feasible synthesis path to resolve a contradiction using A* search.

        The A* search aims to find a path that minimizes 'tension' (cost) while considering
        a heuristic to guide the search towards the target.
        """
        if not self.graph.has_node(source_node) or not self.graph.has_node(target_node):
            return {"error": "Source or target node not in the graph."}

        def heuristic(u, v):
            # Simple heuristic: Euclidean distance if nodes have 'pos' attributes,
            # otherwise a constant (e.g., 1) to make it equivalent to Dijkstra.
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            if 'pos' in u_data and 'pos' in v_data:
                pos_u = u_data['pos']
                pos_v = v_data['pos']
                return ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5
            return 1 # Default heuristic if no positional data

        def edge_cost(u, v, data):
            # Combine tension and feasibility_cost for the edge weight
            return data.get('tension', 0.0) + data.get('feasibility_cost', 0.0)

        try:
            # Use the combined edge_cost for the A* search
            path = nx.astar_path(self.graph, source_node, target_node, heuristic=heuristic, weight=edge_cost)
            path_cost = nx.astar_path_length(self.graph, source_node, target_node, heuristic=heuristic, weight=edge_cost)

            plan = {
                "actions": [f"Synthesize between {path[i]} and {path[i+1]} (tension: {self.graph[path[i]][path[i+1]].get('tension', 'N/A')})" for i in range(len(path)-1)],
                "estimated_tension_reduction": path_cost # The total tension along the path
            }
            return plan
        except nx.NetworkXNoPath:
            return {"error": f"No synthesis path found between {source_node} and {target_node}"}
        except Exception as e:
            return {"error": f"An error occurred during pathfinding: {e}"}

    def evaluate_praxis_outcome(self, plan: SynthesisPlan, feedback: FeedbackLog) -> ResolutionScore:
        """
        Evaluates the outcome of a synthesis plan after execution using multi-objective rewards.

        Args:
            plan (SynthesisPlan): The plan that was executed.
            feedback (FeedbackLog): A dictionary containing feedback metrics, expected to include:
                                    - 'labor_time_saved': float (e.g., percentage or absolute reduction)
                                    - 'energy_efficiency_gain': float (e.g., percentage or absolute improvement)
                                    - 'class_alignment_score': float (e.g., 0.0 to 1.0)
                                    - 'success': bool (overall success indicator)

        Returns:
            ResolutionScore: A score representing the overall success of the resolution (0.0 to 1.0).
        """
        labor_time_saved = feedback.get("labor_time_saved", 0.0)
        energy_efficiency_gain = feedback.get("energy_efficiency_gain", 0.0)
        class_alignment_score = feedback.get("class_alignment_score", 0.0)
        overall_success = feedback.get("success", False)

        # Normalize individual scores if they are not already in a 0-1 range.
        # For simplicity, assuming they are already somewhat normalized or scaled appropriately.
        # A more robust system would have explicit normalization functions.

        # Calculate weighted sum
        weighted_score = (
            self.labor_time_weight * labor_time_saved +
            self.energy_efficiency_weight * energy_efficiency_gain +
            self.class_alignment_weight * class_alignment_score
        )

        # Apply a penalty or bonus based on overall_success
        if not overall_success:
            weighted_score *= 0.5 # Penalize if not successful

        # Ensure the final score is within a reasonable range, e.g., 0 to 1
        # This might require tuning based on the expected range of input metrics.
        # For now, a simple clamping or scaling.
        resolution_score = max(0.0, min(1.0, weighted_score))

        return resolution_score

    def get_graph(self) -> ContradictionGraph:
        """
        Returns the current state of the Contradiction Graph.
        """
        return self.graph

    def update_tension(self, node1: Any, node2: Any, conflict_signal: float):
        """
        Updates the tension of an edge using Exponentially Weighted Moving Average (EWMA).
        A higher conflict_signal indicates more tension.
        """
        if not self.graph.has_edge(node1, node2):
            # If edge doesn't exist, create it with initial tension as conflict_signal
            self.graph.add_edge(node1, node2, tension=conflict_signal, history=[conflict_signal])
            return

        edge_data = self.graph[node1][node2]
        current_tension = edge_data.get("tension", 0.0)
        history = edge_data.get("history", [])

        # Apply EWMA formula
        new_tension = self.ewma_alpha * conflict_signal + (1 - self.ewma_alpha) * current_tension

        edge_data["tension"] = new_tension
        history.append(conflict_signal)
        # Keep history to a reasonable length if needed, e.g., last N signals
        # edge_data["history"] = history[-10:] # Example: keep last 10
        edge_data["history"] = history
