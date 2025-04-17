from typing import Dict, Literal, Optional, Union
from ._digraph_group_chat import DiGraph, DiGraphNode, DiGraphEdge
from autogen_agentchat.agents import BaseChatAgent


class AGGraphBuilder:
    """
    A builder for constructing DiGraph objects programmatically using a fluent API.
    This builder mirrors the structure of LangGraph's GraphBuilder but emits a DiGraph
    compatible with the DiGraphGroupChat system.
    """

    def __init__(self):
        self.nodes: Dict[str, DiGraphNode] = {}
        self.agents: Dict[str, BaseChatAgent] = {}
        self._default_start_node: Optional[str] = None

    def _get_name(self, obj: Union[str, BaseChatAgent]) -> str:
        return obj if isinstance(obj, str) else obj.name

    def add_node(self, agent: BaseChatAgent, activation: Literal["all", "any"] = "all") -> "AGGraphBuilder":
        """Add a node to the graph and register its agent."""
        name = agent.name
        if name not in self.nodes:
            self.nodes[name] = DiGraphNode(name=name, edges=[], activation=activation)
            self.agents[name] = agent
        return self

    def add_edge(
        self,
        source: Union[str, BaseChatAgent],
        target: Union[str, BaseChatAgent],
        condition: Optional[str] = None
    ) -> "AGGraphBuilder":
        """Add a directed edge from source to target, optionally with a condition."""
        source_name = self._get_name(source)
        target_name = self._get_name(target)

        if source_name not in self.nodes:
            raise ValueError(f"Source node '{source_name}' must be added before adding an edge.")
        if target_name not in self.nodes:
            raise ValueError(f"Target node '{target_name}' must be added before adding an edge.")

        self.nodes[source_name].edges.append(DiGraphEdge(target=target_name, condition=condition))
        return self

    def add_conditional_edges(
        self,
        source: Union[str, BaseChatAgent],
        condition_to_target: Dict[str, Union[str, BaseChatAgent]]
    ) -> "AGGraphBuilder":
        """Add multiple conditional edges from a source node based on condition strings."""
        for condition, target in condition_to_target.items():
            self.add_edge(source, target, condition)
        return self

    def set_entry_point(self, name: Union[str, BaseChatAgent]) -> "AGGraphBuilder":
        """Set the default start node of the graph."""
        node_name = self._get_name(name)
        if node_name not in self.nodes:
            raise ValueError(f"Start node '{node_name}' must be added before setting as entry point.")
        self._default_start_node = node_name
        return self

    def build(self) -> DiGraph:
        """Build and validate the DiGraph."""
        graph = DiGraph(
            nodes=self.nodes,
            default_start_node=self._default_start_node,
        )
        graph.validate()
        return graph

    def get_participants(self) -> list[BaseChatAgent]:
        """Return the list of agents in the builder, in insertion order."""
        return list(self.agents.values())
