# parallel_examples.py

import asyncio

from autogen_ext.models.replay import ReplayChatCompletionClient
# from autogen_ext.models.openai import OpenAIChatCompletionClient  # Uncomment to use real OpenAI model
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

from autogen_graph import DiGraph, DiGraphNode, DiGraphEdge, DiGraphGroupChat

# This file includes 3 examples:
# 1. Fan-out: A → B, A → C (parallel instructions)
# 2. Join-any: A → B and A → C, both feeding into D (only one needed to proceed)
# 3. Join-all: A → B and A → C, both feeding into D (both must complete before D starts)

def fan_out_example():
    model = ReplayChatCompletionClient(chat_completions=[
        "Manager: here's a task.",
        "Design: here's a layout sketch.",
        "Dev: here's a prototype implementation.",
    ])
    # model = OpenAIChatCompletionClient(model="gpt-4o")

    manager = AssistantAgent("Manager", model_client=model, system_message="You are assigning a task to a designer and a developer.")
    designer = AssistantAgent("Designer", model_client=model, system_message="Design the solution.")
    developer = AssistantAgent("Developer", model_client=model, system_message="Implement the solution.")

    graph = DiGraph(
        nodes={
            "Manager": DiGraphNode(name="Manager", edges=[DiGraphEdge(target="Designer"), DiGraphEdge(target="Developer")]),
            "Designer": DiGraphNode(name="Designer", edges=[]),
            "Developer": DiGraphNode(name="Developer", edges=[]),
        },
    )

    team = DiGraphGroupChat(
        participants=[manager, designer, developer],
        graph=graph,
        termination_condition=MaxMessageTermination(5),
    )

    return team.run_stream(task="Create a landing page for a weather app")

def join_any_example():
    model = ReplayChatCompletionClient(chat_completions=[
        "Path a and path b start searching.",
        "Found result via path B.",
        "Found result via path C.",
        "Great, result confirmed."
    ])
    # model = OpenAIChatCompletionClient(model="gpt-4o")

    coordinator = AssistantAgent("Coordinator", model_client=model, system_message="Coordinate two paths to explore.")
    path_b = AssistantAgent("PathB", model_client=model, system_message="Search using strategy B.")
    path_c = AssistantAgent("PathC", model_client=model, system_message="Search using strategy C.")
    resolver = AssistantAgent("Resolver", model_client=model, system_message="Wait for any path to return a result and confirm.")

    graph = DiGraph(
        nodes={
            "Coordinator": DiGraphNode(name="Coordinator", edges=[DiGraphEdge(target="PathB"), DiGraphEdge(target="PathC")]),
            "PathB": DiGraphNode(name="PathB", edges=[DiGraphEdge(target="Resolver")]),
            "PathC": DiGraphNode(name="PathC", edges=[DiGraphEdge(target="Resolver")]),
            "Resolver": DiGraphNode(name="Resolver", edges=[], activation="any"),
        },
    )

    team = DiGraphGroupChat(
        participants=[coordinator, path_b, path_c, resolver],
        graph=graph,
        termination_condition=MaxMessageTermination(5),
    )

    return team.run_stream(task="Find the capital of Australia")

def join_all_example():
    model = ReplayChatCompletionClient(chat_completions=[
        "Start path a and path b. Combine both before starting the report.",
        "Research result from path A.",
        "Research result from path B.",
        "Final report combining both results."
    ])
    # model = OpenAIChatCompletionClient(model="gpt-4o")

    lead = AssistantAgent("Lead", model_client=model, system_message="Delegate research to two teams.")
    research_a = AssistantAgent("ResearchA", model_client=model, system_message="Research using method A.")
    research_b = AssistantAgent("ResearchB", model_client=model, system_message="Research using method B.")
    reporter = AssistantAgent("Reporter", model_client=model, system_message="Wait for both research paths before compiling report.")

    graph = DiGraph(
        nodes={
            "Lead": DiGraphNode(name="Lead", edges=[DiGraphEdge(target="ResearchA"), DiGraphEdge(target="ResearchB")]),
            "ResearchA": DiGraphNode(name="ResearchA", edges=[DiGraphEdge(target="Reporter")]),
            "ResearchB": DiGraphNode(name="ResearchB", edges=[DiGraphEdge(target="Reporter")]),
            "Reporter": DiGraphNode(name="Reporter", edges=[], activation="all"),
        },
    )

    team = DiGraphGroupChat(
        participants=[lead, research_a, research_b, reporter],
        graph=graph,
        termination_condition=MaxMessageTermination(5),
    )

    return team.run_stream(task="Summarize findings on climate change from two sources")

async def main():
    print("\n=== Fan-out Example ===\n")
    await Console(fan_out_example())

    print("\n=== Join-Any Example ===\n")
    await Console(join_any_example())

    print("\n=== Join-All Example ===\n")
    await Console(join_all_example())

if __name__ == "__main__":
    asyncio.run(main())
