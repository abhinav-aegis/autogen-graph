# autogen-graph

**Directed Graph-based execution engine for Autogen agents.**

`autogen-graph` allows you to coordinate multi-agent interactions using a flexible, cycle-aware graph structure. It is designed for modeling complex flows, including sequential chains, loops, conditionals, and fan-out/join patterns, using [Autogen](https://github.com/microsoft/autogen)-compatible agents.

---

## 💡 Why Graph-Based Workflows?

Autogen currently provides powerful abstractions for team-based agent interaction via group chats. However, its default broadcast-style message flow lacks precise control over agent execution order, branching, and routing.

A graph-based execution model brings:

- **Explicit control over execution flow**: You define exactly what runs when and where.
- **Agent context isolation**: Each node is an independent unit that takes one input and emits one output, improving predictability and modularity.
- **Reusability and clarity**: Each node can encapsulate logic (including group chats) and expose only the final output.
- **Support for loops, conditions, and escalations**: Graph edges can dynamically route based on message content.
- **Extensibility**: Enables hybrid orchestration models combining autonomy, rule-based flow, and parallelism.

This aligns with future directions proposed by the Autogen team ([issue #4623](https://github.com/microsoft/autogen/issues/4623)), creating a middle ground between full autonomy and deterministic workflow execution.

---

## ✨ Features

- 🔁 Supports agent loops, cycles, and feedback workflows
- 🔀 Execute parallel fan-outs, join-any/join-all, and content-based branching
- 🧩 Seamlessly integrates with `AssistantAgent`, `GroupChat`, and Autogen runtimes
- 🧪 Easily testable using `ReplayChatCompletionClient`
- 🖥 CLI-friendly with built-in `Console` streaming

---

## 📦 Installation

```bash
pip install autogen-graph
```

---

## 🚀 Quickstart

```python
import asyncio

from autogen_graph import DiGraph, DiGraphNode, DiGraphEdge, DiGraphGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create OpenAI-backed agents
model_client = OpenAIChatCompletionClient(model="gpt-4o")

poet = AssistantAgent(
    name="poet",
    model_client=model_client,
    system_message="Write a poem about the ocean."
)

critic = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="Give feedback on the poem. Respond with 'APPROVE' if it's good, otherwise explain what to improve."
)

improver = AssistantAgent(
    name="improve",
    model_client=model_client,
    system_message="Improve the poem based on the critic's feedback."
)

# Define a graph: poet → critic → improve
graph = DiGraph(
    nodes={
        "poet": DiGraphNode(name="poet", edges=[DiGraphEdge(target="critic")]),
        "critic": DiGraphNode(name="critic", edges=[DiGraphEdge(target="improve")]),
        "improve": DiGraphNode(name="improve", edges=[]),
    },
    default_start_node="poet"
)

team = DiGraphGroupChat(
    participants=[poet, critic, improver],
    graph=graph,
    termination_condition=TextMentionTermination("APPROVE"),
)

async def main():
    await Console(team.run_stream("Please write a poem about the ocean."))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📁 Project Structure

```
├── LICENSE.md
├── README.md
├── examples
│   ├── conditional.py
│   ├── loop.py
│   ├── parallel.py
│   └── sequential.py
├── main.py
├── pyproject.toml
├── src
│   └── autogen_graph
│       ├── __init__.py
│       └── _digraph_group_chat.py
├── tests
│   └── test_digraph_group_chat.py
└── uv.lock
```

---

## 🔍 Core Components

- **DiGraph** – Encodes node-to-node agent execution paths (allows cycles)
- **DiGraphNode** – Represents an agent + outgoing edges
- **DiGraphEdge** – Supports optional `condition` for dynamic routing
- **DiGraphGroupChat** – Executes the graph using agent runtime (optionally threaded)

---

## 🧪 Examples

Available in `examples/`:
- `sequential.py` – A → B → C
- `parallel.py` – fan-out, join-any, join-all
- `conditional.py` – conditional branching using content triggers
- `loop.py` – loops and escalation workflows

Run with:
```bash
python examples/loop.py
```

---

## ✅ Running Tests

```bash
pytest tests/
```

---

## 📜 License

MIT © A Somaraju

---

## 🙌 Contributions

We welcome PRs! Especially around:
- graph visualization utilities
- debugging and trace visualization
- new edge activation strategies or runtime policies

---

## 🗂️ TODO

- [ ] Build a fluent API to simplify graph construction (e.g., `graph.add_node(...).connect(...)` chaining)
- [ ] Add examples for handling structured messages, including conditional edge routing based on message fields and source filtering (e.g., only forward messages from specific agents)


