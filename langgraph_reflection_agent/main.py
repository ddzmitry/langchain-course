from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from chains import generate_chain, reflection_chain
from langgraph.graph.message import (
    add_messages,
)  # to add messages to the state graph abstract function
from typing import TypedDict, Annotated

load_dotenv()


class MessageGraph(TypedDict):
    # Since each node will return its own message list, we need to aggregate them use add_messages to do that
    # Define messages as an annotated list of BaseMessage with add_messages decorator
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"
# Function to determine the next step based on the last message
LAST = -1


def generation_node(state: MessageGraph) -> MessageGraph:
    """
    Generation node to create or improve a tweet.
    """
    # Invoke the generation chain with the current messages
    response = generate_chain.invoke({"messages": state["messages"]})
    # Print the response for debugging
    print("Generation Response:", response)
    return {"messages": response}


def reflection_node(state: MessageGraph) -> MessageGraph:
    """
    Reflection node to provide feedback on the tweet.
    """
    # Invoke the reflection chain with the current messages
    response = reflection_chain.invoke({"messages": state["messages"]})
    # Print the response for debugging
    print("Reflection Response:", response)
    # But. here we need to return HumanMessage, so we extract messages from response [Human,AI,Human,AI,...] we have to do that
    # manually since LLM returns AI message and our job here is to return HumanMessage with feedback
    return {"messages": [HumanMessage(content=response.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


# define should_continue function
def should_continue(state: MessageGraph) -> str:
    # Check if the last message contains critique for reflection
    if len(state["messages"]) > 6:
        return END
    return REFLECT


# Add conditional edges
builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
# Define the edges between nodes
builder.add_edge(REFLECT, GENERATE)
# builder.add_edge(GENERATE, REFLECT)
builder.add_edge(GENERATE, END)
graph = builder.compile()
# Draw the flow diagram to a PNG file
graph.get_graph().draw_mermaid_png(
    output_file_path=os.path.join("./langgraph_reflection_agent/reflection_flow.png")
)


# langgraph_reflection_agent
def main():
    print("Hello from langgraph-reflection-agent!")
    res = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Create an engaging tweet about the benefits of having 4 cats."
                )
            ]
        }
    )
    print(res["messages"][LAST].content)

    """
    🐾✨ Why have one cat when you can have FOUR? 🐱🐱🐱🐱 

1. Enjoy endless cuddles & purrs 🥰
2. Built-in entertainment squad 🎭
3. Unique personalities = daily surprises! 🎉
4. Teamwork makes playtime a blast! 💥

Life is better with a feline family! How many cats do you have? Share your favorite cat moments! 🐾❤️ #CatLife #FelineFriends #CatLovers #CatsOfTwitter
    """


if __name__ == "__main__":
    main()
