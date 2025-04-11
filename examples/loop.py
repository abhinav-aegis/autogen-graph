import asyncio

from autogen_ext.models.replay import ReplayChatCompletionClient
# from autogen_ext.models.openai import OpenAIChatCompletionClient  # Uncomment to use real OpenAI model
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

from autogen_graph import DiGraph, DiGraphNode, DiGraphEdge, DiGraphGroupChat

def loop_with_escalation_example():
    model = ReplayChatCompletionClient(chat_completions=[
        "Here's my first draft.",  # Agent
        "Needs improvement. revise",     # Critic 1
        "I've improved it.",      # Agent again
        "Still not there. revise",       # Critic 2
        "This is the final version.", # Agent escalates
        "This is acceptable. review.",  # Critic 3
        "This is acceptable. APPROVED."  # Reviewer (stops loop)
    ])
    # model = OpenAIChatCompletionClient(model="gpt-4o")

    agent = AssistantAgent("Agent", model_client=model, system_message="Respond to a request and revise if needed.")
    critic = AssistantAgent("Critic", model_client=model, system_message="Critique and give feedback. Respond with 'escalate' to trigger escalation.")
    reviewer = AssistantAgent("Reviewer", model_client=model, system_message="Final escalation reviewer. Respond with 'APPROVED' if satisfied.")

    graph = DiGraph(
        nodes={
            "Agent": DiGraphNode(name="Agent", edges=[DiGraphEdge(target="Critic")]),
            "Critic": DiGraphNode(name="Critic", edges=[
                DiGraphEdge(target="Agent", condition="revise"),
                DiGraphEdge(target="Reviewer", condition="review")
            ]),
            "Reviewer": DiGraphNode(name="Reviewer", edges=[]),
        },
        default_start_node="Agent"
    )

    team = DiGraphGroupChat(
        participants=[agent, critic, reviewer],
        graph=graph,
        termination_condition=MaxMessageTermination(10),
    )

    return team.run_stream(task="Write a short bio about Alan Turing.")

async def main():
    print("\n=== Loop with Escalation Example ===\n")
    await Console(loop_with_escalation_example())

if __name__ == "__main__":
    asyncio.run(main())