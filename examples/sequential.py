
import asyncio

from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient # type: ignore
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

from autogen_graph import DiGraph, DiGraphNode, DiGraphGroupChat, DiGraphEdge

async def main():
    # Define a model client. You can use other model client that implements
    # the `ChatCompletionClient` interface.
    # model_client = OpenAIChatCompletionClient(
    #     model="gpt-4o-mini",
    #     # api_key="your_api_key_here",
    # )

    model_client = ReplayChatCompletionClient(
        chat_completions=[
            "here is a nice poem",
            "the poem is not nice",
            "Sure, here is a better poem",
        ]
    )

    primary = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent.
    critic = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    improve = AssistantAgent(
        "improve",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    graph = DiGraph(
        nodes={
            "primary": DiGraphNode(name="primary", edges=[DiGraphEdge(target="critic")]),
            "critic": DiGraphNode(name="critic", edges=[DiGraphEdge(target="improve")]),
            "improve": DiGraphNode(name="improve", edges=[]),
        }
    )

    team = DiGraphGroupChat(
        participants=[primary, critic, improve],
        graph=graph,
        termination_condition=MaxMessageTermination(5),
    )
    
    await Console(team.run_stream(task="Write me a funny poem about winter."))

if __name__ == "__main__":
    asyncio.run(main())
    
