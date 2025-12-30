from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
# ACT is the node that will run the tools
ACT = "act"
# Define constant for accessing the last message
LAST = -1
# Function to determine whether to continue or end the flow


# based on whether there are tool calls in the last message
def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


#   Create the state graph
flow = StateGraph(MessagesState)
#   Add nodes to the graph
flow.add_node(AGENT_REASON, run_agent_reasoning)
# Set the entry point of the graph
flow.set_entry_point(AGENT_REASON)
#   Add the tool node
flow.add_node(ACT, tool_node)
#  Add conditional edges based on whether to continue or end
flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
#  Add edge from ACT back to AGENT_REASON to continue the loop
flow.add_edge(ACT, AGENT_REASON)
# Compile the flow into an executable app
app = flow.compile()
# Draw the flow diagram to a PNG file
app.get_graph().draw_mermaid_png(output_file_path="./app/flow.png")
# Run the app with a sample input
if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the temperature in Minsk? List it and then triple it, display all the results in farenhite."
                )
            ]
        }
    )
    print(res["messages"][LAST].content)
