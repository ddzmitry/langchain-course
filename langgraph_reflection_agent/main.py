from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
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
    score: float  # This will hold the score of the tweet


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
    Reflection node to provide feedback on the tweet and extract score.
    """
    # Invoke the reflection chain with the current messages (returns ReflectionOutput)
    response = reflection_chain.invoke({"messages": state["messages"]})
    # Print the response for debugging
    print(
        f"Reflection Response - Score: {response.score}, Feedback: {response.feedback}"
    )
    # Return HumanMessage with feedback and update the score in state
    return {
        "messages": [HumanMessage(content=response.feedback)],
        "score": response.score,
    }


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


# define should_continue function
def should_continue(state: MessageGraph) -> str:
    """
    Determine whether to continue reflection or end.
    Ends if: message count > 6 OR score > 9
    """
    # Check if we've exceeded the maximum message count
    if len(state["messages"]) > 6:
        print("Ending: Maximum message count reached")
        return END

    # Check if we have a score and if it's above 9
    if state.get("score", 0) > 9:
        print(f"Ending early: Score {state['score']} is above 9")
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
# graph.get_graph().draw_mermaid_png(
#     output_file_path=os.path.join("./langgraph_reflection_agent/reflection_flow.png")
# )


# langgraph_reflection_agent
def main():
    print("Hello from langgraph-reflection-agent!")
    res = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Create an engaging tweet about the benefits of having 4 cats."
                )
            ],
            "score": 0.0,  # Initialize score
        }
    )
    print(f"\nFinal Score: {res.get('score', 'N/A')}")
    print(f"\nFinal Tweet:\n{res['messages'][LAST].content}")

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
