"""
Contradiction Engine for SAAF-OS

This module implements the Contradiction Engine as specified in ContradictionEngine.md.
It detects, models, and resolves dialectical contradictions in goals, plans, and world-states
to guide synthesis rather than binary conflict resolution.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
import heapq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContradictionEngine")

class Node:
    """Represents a node in the Contradiction Graph (goal, value, agent, resource, or social group)."""
    
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """
        Initialize a node.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (goal, value, agent, resource, social_group)
            properties: Dictionary of node properties
        """
        self.id = node_id
        self.type = node_type
        self.properties = properties
    
    def __str__(self) -> str:
        """String representation of the node."""
        return f"Node({self.id}, {self.type})"
    
    def __repr__(self) -> str:
        """Formal string representation of the node."""
        return self.__str__()


class Edge:
    """Represents a contradiction edge in the Contradiction Graph."""
    
    def __init__(self, edge_id: str, source_id: str, target_id: str, 
                 contradiction_type: str, tension: float = 0.0):
        """
        Initialize an edge.
        
        Args:
            edge_id: Unique identifier for the edge
            source_id: ID of the source node
            target_id: ID of the target node
            contradiction_type: Type of contradiction (antagonistic or non-antagonistic)
            tension: Tension score [0-1]
        """
        self.id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.contradiction_type = contradiction_type  # antagonistic or non-antagonistic
        self.tension = tension
        self.history = []  # List of historical tension values
    
    def update_tension(self, new_tension: float, alpha: float = 0.3) -> None:
        """
        Update tension score using exponentially weighted moving average (EWMA).
        
        Args:
            new_tension: New tension value
            alpha: EWMA smoothing factor
        """
        # Store current tension in history
        self.history.append(self.tension)
        
        # Update tension with EWMA
        if len(self.history) > 1:
            self.tension = alpha * new_tension + (1 - alpha) * self.tension
        else:
            self.tension = new_tension
    
    def __str__(self) -> str:
        """String representation of the edge."""
        return f"Edge({self.id}, {self.source_id}->{self.target_id}, {self.contradiction_type}, tension={self.tension:.2f})"
    
    def __repr__(self) -> str:
        """Formal string representation of the edge."""
        return self.__str__()


class ContradictionGraph:
    """
    Implementation of the Contradiction Graph (CG) as specified in ContradictionEngine.md.
    Uses NetworkX for the underlying graph representation.
    """
    
    def __init__(self):
        """Initialize an empty contradiction graph."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
    
    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: The node to add
        """
        self.nodes[node.id] = node
        self.graph.add_node(node.id, type=node.type, properties=node.properties)
    
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the graph.
        
        Args:
            edge: The edge to add
        """
        self.edges[edge.id] = edge
        self.graph.add_edge(
            edge.source_id, 
            edge.target_id, 
            id=edge.id,
            contradiction_type=edge.contradiction_type,
            tension=edge.tension,
            history=edge.history
        )
    
    def update_edge_tension(self, edge_id: str, new_tension: float) -> None:
        """
        Update the tension of an edge.
        
        Args:
            edge_id: ID of the edge to update
            new_tension: New tension value
        """
        if edge_id in self.edges:
            self.edges[edge_id].update_tension(new_tension)
            
            # Update the edge in the NetworkX graph as well
            source_id = self.edges[edge_id].source_id
            target_id = self.edges[edge_id].target_id
            self.graph[source_id][target_id]['tension'] = self.edges[edge_id].tension
            self.graph[source_id][target_id]['history'] = self.edges[edge_id].history
    
    def get_high_tension_edges(self, threshold: float = 0.5) -> List[Edge]:
        """
        Get edges with tension above the threshold.
        
        Args:
            threshold: Tension threshold
            
        Returns:
            List of edges with tension above the threshold
        """
        return [edge for edge in self.edges.values() if edge.tension > threshold]
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_edge_between(self, source_id: str, target_id: str) -> Optional[Edge]:
        """Get the edge between two nodes if it exists."""
        if self.graph.has_edge(source_id, target_id):
            edge_id = self.graph[source_id][target_id]['id']
            return self.edges.get(edge_id)
        return None


class SynthesisPlan:
    """
    Represents a plan to resolve contradictions through synthesis.
    """
    
    def __init__(self, plan_id: str):
        """
        Initialize a synthesis plan.
        
        Args:
            plan_id: Unique identifier for the plan
        """
        self.id = plan_id
        self.actions = []
        self.tension_diff = 0.0
        self.synthesis_path = []
    
    def add_action(self, action: Dict[str, Any]) -> None:
        """
        Add an action to the plan.
        
        Args:
            action: Dictionary describing the action
        """
        self.actions.append(action)
    
    def set_synthesis_path(self, path: List[str]) -> None:
        """
        Set the synthesis path (sequence of node transformations).
        
        Args:
            path: List of node IDs in the synthesis path
        """
        self.synthesis_path = path
    
    def set_tension_diff(self, diff: float) -> None:
        """
        Set the expected tension difference after applying the plan.
        
        Args:
            diff: Expected tension difference (should be negative for good plans)
        """
        self.tension_diff = diff
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the plan to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the plan
        """
        return {
            "plan_id": self.id,
            "actions": self.actions,
            "tension_diff": self.tension_diff,
            "synthesis_path": self.synthesis_path
        }


class ASynthesisSearch:
    """
    A* search algorithm for finding synthesis paths in the contradiction graph.
    """
    
    def __init__(self, graph: ContradictionGraph):
        """
        Initialize the A* search.
        
        Args:
            graph: The contradiction graph to search
        """
        self.graph = graph
    
    def heuristic(self, node_id: str, goal_id: str) -> float:
        """
        Heuristic function for A* search: sum of tension divided by feasibility.
        
        Args:
            node_id: Current node ID
            goal_id: Goal node ID
            
        Returns:
            Heuristic value
        """
        # Simple heuristic: Use shortest path distance as a base
        try:
            path_length = nx.shortest_path_length(self.graph.graph, node_id, goal_id)
            return path_length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')
    
    def search(self, start_id: str, goal_id: str) -> Optional[List[str]]:
        """
        Perform A* search to find a synthesis path.
        
        Args:
            start_id: Start node ID
            goal_id: Goal node ID
            
        Returns:
            List of node IDs forming the path, or None if no path found
        """
        if start_id not in self.graph.nodes or goal_id not in self.graph.nodes:
            return None
            
        # Priority queue for A* search
        open_set = []
        heapq.heappush(open_set, (0, start_id))
        
        # Best paths and costs
        came_from = {}
        g_score = {start_id: 0}  # Cost from start
        f_score = {start_id: self.heuristic(start_id, goal_id)}  # Estimated total cost
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            for neighbor in self.graph.graph.neighbors(current):
                # Get the edge between current and neighbor
                edge = self.graph.get_edge_between(current, neighbor)
                if not edge:
                    continue
                    
                # Edge weight is based on tension
                weight = 1 + edge.tension  # Higher tension = higher cost
                
                # Calculate tentative g score
                tentative_g_score = g_score.get(current, float('inf')) + weight
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_id)
                    
                    # Add to open set if not already there
                    if not any(neighbor == node for _, node in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None


class ContradictionEngine:
    """
    Main class implementing the Contradiction Engine as specified in ContradictionEngine.md.
    """
    
    def __init__(self):
        """Initialize the Contradiction Engine."""
        self.contradiction_graph = ContradictionGraph()
        self.synthesis_search = ASynthesisSearch(self.contradiction_graph)
    
    def detect_contradictions(self, goals: Dict[str, Any], world_state: Dict[str, Any]) -> ContradictionGraph:
        """
        Detect contradictions between goals and world state.
        
        Args:
            goals: Dictionary of goals and their properties
            world_state: Dictionary representing the current world state
            
        Returns:
            Updated contradiction graph
        """
        # For each goal, check for resource conflicts
        for goal_id, goal_props in goals.items():
            # Create or update goal node
            if goal_id not in self.contradiction_graph.nodes:
                goal_node = Node(goal_id, "goal", goal_props)
                self.contradiction_graph.add_node(goal_node)
            
            # Check for resource conflicts with other goals
            required_resources = goal_props.get("required_resources", [])
            for resource_id in required_resources:
                # Create resource node if it doesn't exist
                if resource_id not in self.contradiction_graph.nodes:
                    resource_node = Node(
                        resource_id, 
                        "resource", 
                        world_state.get("resources", {}).get(resource_id, {})
                    )
                    self.contradiction_graph.add_node(resource_node)
                
                # Check if this resource is constrained
                resource_state = world_state.get("resources", {}).get(resource_id, {})
                if resource_state.get("constrained", False):
                    # Find other goals that require this resource
                    for other_goal_id, other_goal_props in goals.items():
                        if (other_goal_id != goal_id and 
                            resource_id in other_goal_props.get("required_resources", [])):
                            
                            # Create or get the other goal node
                            if other_goal_id not in self.contradiction_graph.nodes:
                                other_goal_node = Node(other_goal_id, "goal", other_goal_props)
                                self.contradiction_graph.add_node(other_goal_node)
                            
                            # Calculate tension based on resource scarcity
                            scarcity = resource_state.get("scarcity", 0.5)
                            tension = scarcity * 0.8  # Scale to get tension
                            
                            # Create contradiction edge between goals
                            edge_id = f"{goal_id}_vs_{other_goal_id}_{resource_id}"
                            contradiction_type = "antagonistic" if tension > 0.6 else "non-antagonistic"
                            
                            # Check if the edge already exists
                            edge = self.contradiction_graph.get_edge_between(goal_id, other_goal_id)
                            if edge:
                                # Update existing edge tension
                                self.contradiction_graph.update_edge_tension(edge.id, tension)
                            else:
                                # Create new edge
                                edge = Edge(edge_id, goal_id, other_goal_id, contradiction_type, tension)
                                self.contradiction_graph.add_edge(edge)
        
        # Create edges for value contradictions
        for goal_id, goal_props in goals.items():
            for other_goal_id, other_goal_props in goals.items():
                if goal_id != other_goal_id:
                    # Compare values
                    goal_values = set(goal_props.get("values", []))
                    other_values = set(other_goal_props.get("values", []))
                    
                    # Check for opposing values
                    opposing_values = world_state.get("opposing_values", {})
                    
                    # Calculate value contradiction
                    value_tension = 0.0
                    for value in goal_values:
                        for other_value in other_values:
                            if (value in opposing_values and other_value in opposing_values.get(value, [])):
                                value_tension = max(value_tension, 0.7)  # High tension for opposing values
                    
                    if value_tension > 0:
                        edge_id = f"{goal_id}_value_vs_{other_goal_id}"
                        contradiction_type = "antagonistic" if value_tension > 0.6 else "non-antagonistic"
                        
                        # Check if the edge already exists
                        edge = self.contradiction_graph.get_edge_between(goal_id, other_goal_id)
                        if edge:
                            # Update with higher tension if applicable
                            self.contradiction_graph.update_edge_tension(edge.id, max(value_tension, edge.tension))
                        else:
                            # Create new edge
                            edge = Edge(edge_id, goal_id, other_goal_id, contradiction_type, value_tension)
                            self.contradiction_graph.add_edge(edge)
                        
        return self.contradiction_graph
    
    def resolve_contradiction(self, c1_id: str, c2_id: str, strategy: str = "synthesis") -> Optional[SynthesisPlan]:
        """
        Resolve a contradiction between two nodes.
        
        Args:
            c1_id: ID of the first contradicting node
            c2_id: ID of the second contradicting node
            strategy: Resolution strategy, defaults to "synthesis"
            
        Returns:
            Synthesis plan or None if no resolution found
        """
        if strategy == "synthesis":
            # Find a synthesis path using A* search
            path = self.synthesis_search.search(c1_id, c2_id)
            if not path:
                logger.warning(f"Failed to find synthesis path between {c1_id} and {c2_id}")
                return None
            
            # Create a synthesis plan
            plan = SynthesisPlan(f"synthesis_plan_{c1_id}_{c2_id}")
            plan.set_synthesis_path(path)
            
            # Calculate expected tension reduction
            edge = self.contradiction_graph.get_edge_between(c1_id, c2_id)
            if edge:
                plan.set_tension_diff(-edge.tension * 0.8)  # Assume 80% reduction
            
            # Generate actions based on nodes in the path
            for i in range(len(path) - 1):
                source_id = path[i]
                target_id = path[i + 1]
                edge = self.contradiction_graph.get_edge_between(source_id, target_id)
                
                if edge:
                    action = self._generate_action_for_edge(edge)
                    plan.add_action(action)
            
            return plan
        else:
            # Other strategies could be implemented here
            logger.warning(f"Strategy {strategy} not implemented")
            return None
    
    def _generate_action_for_edge(self, edge: Edge) -> Dict[str, Any]:
        """
        Generate an action to resolve the contradiction represented by an edge.
        
        Args:
            edge: The contradiction edge
            
        Returns:
            Action dictionary
        """
        source_node = self.contradiction_graph.get_node(edge.source_id)
        target_node = self.contradiction_graph.get_node(edge.target_id)
        
        if not source_node or not target_node:
            return {"type": "invalid", "description": "Invalid nodes"}
        
        # Generate action based on node types
        if source_node.type == "goal" and target_node.type == "goal":
            # Goal conflict action
            return {
                "type": "redistribute_resources",
                "description": f"Redistribute resources between {source_node.id} and {target_node.id}",
                "source_goal": source_node.id,
                "target_goal": target_node.id,
                "expected_tension_reduction": edge.tension * 0.5
            }
        elif source_node.type == "resource" or target_node.type == "resource":
            # Resource constraint action
            return {
                "type": "optimize_resource",
                "description": f"Optimize resource usage for {source_node.id if source_node.type == 'resource' else target_node.id}",
                "resource_id": source_node.id if source_node.type == "resource" else target_node.id,
                "goal_id": target_node.id if source_node.type == "resource" else source_node.id,
                "expected_tension_reduction": edge.tension * 0.6
            }
        else:
            # Default action
            return {
                "type": "mediate",
                "description": f"Mediate contradiction between {source_node.id} and {target_node.id}",
                "source_id": source_node.id,
                "target_id": target_node.id,
                "expected_tension_reduction": edge.tension * 0.3
            }
    
    def evaluate_praxis_outcome(self, plan: SynthesisPlan, feedback: Dict[str, Any]) -> float:
        """
        Evaluate the outcome of applying a synthesis plan.
        
        Args:
            plan: The synthesis plan that was applied
            feedback: Feedback data on the outcome
            
        Returns:
            Resolution score [0-1], higher is better
        """
        # Extract feedback metrics
        success_rate = feedback.get("success_rate", 0.0)
        tension_reduction = feedback.get("tension_reduction", 0.0)
        labor_time = feedback.get("labor_time", 0.0)
        energy_usage = feedback.get("energy_usage", 0.0)
        
        # Calculate multi-objective score
        # Higher success and tension reduction are better, lower labor time and energy are better
        score = (0.4 * success_rate + 
                0.3 * tension_reduction + 
                0.2 * (1.0 - min(1.0, labor_time / 100.0)) + 
                0.1 * (1.0 - min(1.0, energy_usage / 100.0)))
        
        # Update the graph based on feedback
        for node_id in plan.synthesis_path:
            # Update node properties based on feedback if applicable
            if node_id in feedback.get("node_updates", {}):
                node = self.contradiction_graph.get_node(node_id)
                if node:
                    node.properties.update(feedback["node_updates"][node_id])
        
        # Update edge tensions
        for i in range(len(plan.synthesis_path) - 1):
            source_id = plan.synthesis_path[i]
            target_id = plan.synthesis_path[i + 1]
            edge = self.contradiction_graph.get_edge_between(source_id, target_id)
            
            if edge:
                # Reduce tension based on success rate
                new_tension = edge.tension * (1.0 - success_rate * tension_reduction)
                self.contradiction_graph.update_edge_tension(edge.id, new_tension)
        
        return score