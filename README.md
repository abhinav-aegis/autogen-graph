# autogen-graph

**Directed Graph-based execution engine for Autogen agents, with optional message filtering.**

`autogen-graph` lets you design deterministic, conditional, and cyclic workflows between Autogen-compatible agents. It supports both *graph-based execution control* and *message filtering* to precisely govern **when** agents run and **what messages** they see.

---

## 💡 What Does This Provide?

Autogen’s default group chats use a broadcast model. While powerful, it lacks precision:

- Agents can't be triggered conditionally.
- Message history grows without control.
- Parallelism and loops require manual workarounds.

`autogen-graph` solves this by introducing:

### 🔹 1. Graph-Based Execution (DiGraph)
Define **who runs next** using a fluent API to build nodes and edges.

- Control execution order
- Support parallel fan-outs, joins, conditionals
- Handle loops with runtime-safe cycles

### 🔹 2. Message Filtering (`MessageFilterAgent`)
Control **what messages each agent sees** before they're invoked.

- Restrict to last N messages from a source
- Include only specific message types or senders
- Prevent irrelevant context from leaking

This decouples execution routing from message visibility.

---

## ✨ Features

- ✅ Directed graph with support for:
  - ⏩ Sequential flows
  - 🔀 Parallel branches and joins
  - ♻️ Loops with runtime-safe cycles
  - ❓ Conditional edge activation
- 🧹 `MessageFilterAgent` to control per-agent context
- 🧪 Test-friendly with `ReplayChatCompletionClient`
- 📎 CLI-friendly with `Console` streaming

---

## 🗖️ Quickstart: Graph-based Flow

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_graph import DiGraphGroupChat, AGGraphBuilder

import asyncio

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define agents
poet = AssistantAgent(name="poet", model_client=model_client, system_message="Write a poem about the ocean.")
critic = AssistantAgent(name="critic", model_client=model_client, system_message="Critique the poem and say APPROVE or revise.")
improver = AssistantAgent(name="improve", model_client=model_client, system_message="Improve the poem.")

# Build the execution graph
builder = AGGraphBuilder()
builder.add_node(poet).add_node(critic).add_node(improver)
builder.add_edge(poet, critic).add_edge(critic, improver)

# Assemble team
team = DiGraphGroupChat(
    participants=builder.get_participants(),
    graph=builder.build(),
    termination_condition=TextMentionTermination("APPROVE"),
)

async def main():
    await Console(team.run_stream("Please write a poem about the ocean."))

asyncio.run(main())
```

---

## 🔍 Message Filtering Example

Use `MessageFilterAgent` to restrict what messages an agent receives:

```python
from autogen_graph import MessageFilterAgent, MessageFilterConfig, PerSourceFilter

filtered_critic = MessageFilterAgent(
    name="critic",
    wrapped_agent=critic,
    filter=MessageFilterConfig(
        per_source=[
            PerSourceFilter(source="poet", position="last", count=1),      # only last poet message
            PerSourceFilter(source="user", position="first", count=1),     # only first user message
        ]
    )
)

builder = AGGraphBuilder()
builder.add_node(poet).add_node(filtered_critic).add_node(improver)
builder.add_edge(poet, "critic").add_edge("critic", improver)

team = DiGraphGroupChat(
    participants=builder.get_participants(),
    graph=builder.build(),
    termination_condition=TextMentionTermination("APPROVE"),
)
```

This ensures `critic` only sees the last message from `poet` and the first message from `user`.

---

## 🔁 Advanced Example: Conditional Loop + Filtered Summary

This example demonstrates:
- A loop between `generator` and `reviewer` (which exits when reviewer says "final")
- A `summarizer` agent that only sees the **first user input** and the **last reviewer message**

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_graph import (
    DiGraphGroupChat, AGGraphBuilder,
    MessageFilterAgent, MessageFilterConfig, PerSourceFilter,
)

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Agents
generator = AssistantAgent("generator", model_client=model_client, system_message="Generate a list of creative ideas.")
reviewer = AssistantAgent("reviewer", model_client=model_client, system_message="Review ideas and say LOOP or FINAL.")
summarizer_core = AssistantAgent("summary", model_client=model_client, system_message="Summarize the user request and the final feedback.")

# Filtered summarizer
filtered_summarizer = MessageFilterAgent(
    name="summary",
    wrapped_agent=summarizer_core,
    filter=MessageFilterConfig(
        per_source=[
            PerSourceFilter(source="user", position="first", count=1),
            PerSourceFilter(source="reviewer", position="last", count=1),
        ]
    )
)

# Build graph with conditional loop
builder = AGGraphBuilder()
builder.add_node(generator).add_node(reviewer).add_node(filtered_summarizer)
builder.add_edge(generator, reviewer)
builder.add_edge(reviewer, generator, condition="LOOP")
builder.add_edge(reviewer, filtered_summarizer, condition="FINAL")

team = DiGraphGroupChat(
    participants=builder.get_participants(),
    graph=builder.build(),
    termination_condition=TextMentionTermination("FINAL"),
)

import asyncio

async def main():
    await Console(team.run_stream("Brainstorm ways to reduce plastic waste."))

asyncio.run(main())
```

This allows iteration until the reviewer says "FINAL", at which point the summary is generated with only the original prompt and final feedback.

---

## 🧠 Conceptual Summary

| Concept                | Purpose                                 | Component                     |
|------------------------|------------------------------------------|-------------------------------|
| Execution control      | Decides **when an agent runs**           | `DiGraph`, `DiGraphGroupChat`, `AGGraphBuilder` |
| Context filtering      | Decides **what messages an agent sees**  | `MessageFilterAgent`          |

Both can be combined seamlessly.

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📁 Project Structure

```
src/autogen_graph/
├── _digraph_group_chat.py      # Main graph runner
├── _graph_builder.py           # Fluent graph builder
├── _message_filter_agent.py    # Message filtering agent
├── __init__.py
```

---

## 📜 License

MIT © A Somaraju

---

## 🙌 Contributions

Welcome! Especially around:

- Graph editors or visualizations
- New agent container wrappers (e.g., summarizer)
- Message transformation logic

