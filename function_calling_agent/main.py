# Function Calling Agent Implementation
# This file demonstrates how to build an agent using OpenAI's native function calling
# capabilities with manual message handling to understand the mechanics.

from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool
from langchain_openai import ChatOpenAI

from callback import AgentCallbackHandler

# Load environment variables to make API keys available
load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters.

    Args:
        text: The input text string to measure

    Returns:
        The number of characters in the text after stripping quotes and newlines
    """
    print(f"get_text_length enter with {text=}")
    # Stripping away non-alphabetic characters just in case
    text = text.strip("'\n").strip('"')

    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """Find a tool by its name from a list of tools.

    Args:
        tools: A list of BaseTool objects
        tool_name: The name of the tool to find

    Returns:
        The BaseTool object with the specified name

    Raises:
        ValueError: If no tool with the specified name is found
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello LangChain Tools (.bind_tools)!")

    # Step 1: Define available tools
    # Tools are functions decorated with @tool that the LLM can call
    tools = [get_text_length]

    # Step 2: Initialize the ChatOpenAI model
    # Function calling requires a model that supports it (gpt-3.5-turbo, gpt-4, etc.)
    llm = ChatOpenAI(
        temperature=0,  # Deterministic outputs for consistent behavior
        callbacks=[AgentCallbackHandler()],  # Custom callback for logging
    )

    # Step 3: Bind tools to the LLM using .bind_tools()
    # This is the KEY method that enables function calling
    # It tells the LLM what functions are available and their schemas
    # The LLM can then return structured tool calls instead of just text
    llm_with_tools = llm.bind_tools(tools)

    # Step 4: Start the conversation with an initial human message
    # Messages are the core communication format in function calling
    messages = [HumanMessage(content="What is the length of 'strawberry' in characters?")]

    # Step 5: Main conversation loop
    # This loop handles the back-and-forth between the LLM and tool execution
    while True:
        # Invoke the LLM with the current message history
        # The LLM will either:
        # - Return a final text answer (ai_message.content)
        # - Request tool calls (ai_message.tool_calls)
        ai_message = llm_with_tools.invoke(messages)

        # Step 6: Check if the model wants to call any tools
        # tool_calls is a list of tool invocations the LLM wants to make
        # Each tool_call contains: id, name, args
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if len(tool_calls) > 0:
            # The LLM wants to use tools, so we need to:
            # 1. Add the AI's message (with tool calls) to history
            # 2. Execute each requested tool
            # 3. Add tool results as ToolMessages
            # 4. Continue the loop so LLM can process the results

            messages.append(ai_message)

            for tool_call in tool_calls:
                # Extract tool call details
                # tool_call is a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                # Find and execute the requested tool
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                # Add the tool result back to the conversation
                # ToolMessage connects the result to the original tool_call via tool_call_id
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )

            # Continue loop to allow the model to use the observations
            continue

        # Step 7: No tool calls means we have a final answer
        # The LLM has all the information it needs and is responding with text
        print("\n" + "="*50)
        print("FINAL ANSWER")
        print("="*50)
        print(ai_message.content)
        print("="*50)
        break


# Key Advantages of Function Calling vs ReAct:
#
# 1. STRUCTURED TOOL CALLS:
#    - ReAct: LLM outputs text like "Action: get_text_length\nAction Input: strawberry"
#              which must be parsed with regex/string manipulation
#    - Function Calling: LLM returns structured dict:
#              {"name": "get_text_length", "args": {"text": "strawberry"}, "id": "call_123"}
#              No parsing errors, guaranteed valid format
#
# 2. MESSAGE-BASED CONVERSATION:
#    - ReAct: Single string with "Thought/Observation" markers that grows with each step
#    - Function Calling: Structured message list (HumanMessage, AIMessage, ToolMessage)
#              Each message has a clear role and content
#
# 3. TOOL CALL TRACKING:
#    - ReAct: Must manually track which observation goes with which action
#    - Function Calling: tool_call_id links each ToolMessage to its original request
#              Prevents confusion when multiple tools are called
#
# 4. SIMPLER PROMPTS:
#    - ReAct: Requires detailed format instructions in prompt
#              "Use this format: Question/Thought/Action/Action Input/Observation..."
#    - Function Calling: No format instructions needed
#              The .bind_tools() method handles everything
#
# 5. NATIVE MODEL SUPPORT:
#    - ReAct: Model must learn format through prompt engineering
#    - Function Calling: Models are specifically trained for this pattern
#              More reliable, fewer errors
#
# How .bind_tools() Works:
# - Converts Python function signatures into JSON schemas
# - Sends schemas to the LLM as available "tools"
# - LLM can then invoke these tools by returning structured tool_calls
# - No manual prompt engineering needed for tool descriptions
