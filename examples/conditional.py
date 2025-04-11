# conditional_examples.py

import asyncio

from autogen_ext.models.replay import ReplayChatCompletionClient
# from autogen_ext.models.openai import OpenAIChatCompletionClient  # Uncomment to use real OpenAI model
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

from autogen_graph import DiGraph, DiGraphNode, DiGraphEdge, DiGraphGroupChat

def conditional_branch_example():
    model = ReplayChatCompletionClient(chat_completions=[
        "The weather is rainy today.",  # Agent A
        "I recommend carrying an umbrella.",  # Agent B (condition="rain")
        "Or consider indoor activities instead."  # Agent C (condition="indoor")
    ])
    # model = OpenAIChatCompletionClient(model="gpt-4o")

    agent_a = AssistantAgent("Forecaster", model_client=model, system_message="Provide a weather update.")
    agent_b = AssistantAgent("UmbrellaAdvisor", model_client=model, system_message="Give advice if it's raining.")
    agent_c = AssistantAgent("IndoorPlanner", model_client=model, system_message="Give ideas for indoor activities.")

    graph = DiGraph(
        nodes={
            "Forecaster": DiGraphNode(name="Forecaster", edges=[
                DiGraphEdge(target="UmbrellaAdvisor", condition="rain"),
                DiGraphEdge(target="IndoorPlanner", condition="indoor")
            ]),
            "UmbrellaAdvisor": DiGraphNode(name="UmbrellaAdvisor", edges=[]),
            "IndoorPlanner": DiGraphNode(name="IndoorPlanner", edges=[]),
        },
    )

    team = DiGraphGroupChat(
        participants=[agent_a, agent_b, agent_c],
        graph=graph,
        termination_condition=MaxMessageTermination(5),
    )

    return team.run_stream(task="Tell me if I should bring an umbrella or stay indoors.")

async def main():
    print("\n=== Conditional Branch Example ===\n")
    await Console(conditional_branch_example())

if __name__ == "__main__":
    asyncio.run(main())
