from dotenv import load_dotenv
from langgraph.graph import MessagesState

# Tool node is used by LLM response to call external tools
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]
    )
    print("Agent Response:", response.content)
    return {"messages": [response]}


tool_node = ToolNode(tools)
