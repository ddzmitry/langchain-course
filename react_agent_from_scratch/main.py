# GitHub Copilot
# Detailed, line-by-line documentation for the agent in this file.
# Each comment explains the purpose of the following line(s) concisely.

# Load environment variables from a .env file into the process environment.
from dotenv import load_dotenv

# Decorator to mark a function as a tool usable by the agent.
from langchain.tools import tool

# PromptTemplate builds prompt text with slots and supports partial application.
from langchain_core.prompts import PromptTemplate

# Helper to render a textual description of tools into the prompt.
from langchain_core.tools.render import render_text_description

# OpenAI chat model wrapper provided by langchain.
from langchain_openai import ChatOpenAI

# Parser that converts ReAct-style single-input agent outputs into actions/finishes.
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser

# Type hints for lists, tuples, unions.
from typing import List, Tuple, Union

# AgentAction and AgentFinish represent agent decisions and final outputs.
from langchain_classic.schema import AgentAction, AgentFinish

# Tool type from ollama, used to represent tools in the tools list.
from ollama import Tool

# Actually load environment variables now so API keys and settings are available.
load_dotenv()


# Declare a simple tool that returns the length of a given text.
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    # Debug print showing what input text the tool received.
    print(f"Calculating length of text: {text=}")
    # Strip common quoting and newlines before measuring.
    text = text.strip("'\n").strip('"')
    # Return the number of characters in the cleaned string.
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """Find a tool by its name from a list of tools.
    Args:
        tools: A list of Tool objects.
        tool_name: The name of the tool to find.
    Returns:
        The Tool object with the specified name.
    Raises:
        ValueError: If no tool with the specified name is found.
    """
    # Iterate over the provided tools list.
    for tool in tools:
        # Match the tool by its .name attribute.
        if tool.name == tool_name:
            # Return the matching tool when found.
            return tool
    # Raise a clear error if no matching tool is found.
    raise ValueError(f"Tool with name {tool_name} not found.")


def format_log_to_string(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Formats the intermediate steps into a single string log.
    Args:
        intermediate_steps: A list of tuples containing AgentAction and observation strings.
        observation_prefix: The prefix to use for observations.
        llm_prefix: The prefix to use for LLM thoughts.
    Returns:
        A formatted string representing the log of intermediate steps.
    """
    # Start with an empty accumulator for thought logs.
    thoughts = ""
    # Loop through each (AgentAction, observation) tuple in order.
    for action, observation in intermediate_steps:
        # Append the agent's own log (what it previously thought/sent).
        thoughts += action.log
        # Append the observation header and the raw observation text.
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    # Print the formatted thoughts for debugging/visibility.
    print(f"Formatted thoughts log: {thoughts}")
    # Return the single concatenated string the LLM can consume as context.
    return thoughts


# Standard Python module entrypoint guard.
if __name__ == "__main__":
    # Informational print when running the script directly.
    print("Running the main application...")

    # Build the list of available tools for this agent (only one here).
    tools = [get_text_length]
    # Prompt template that instructs the model how to reason and call tools.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # Render the names and descriptions of tools and partially fill the template.
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    )
    # Prepare an empty list to collect intermediate (action, observation) steps.
    intermediate_steps = []
    # Create a ChatOpenAI LLM instance with deterministic behavior.
    # stop instructs the model to stop when it outputs "Observation" to allow tool handling.
    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation", "Observation"]
    )  # Stop at Observation to allow for multi-step reasoning , Observation is a result of the tool call

    # Build the "chain" / agent pipeline:
    #  - first a mapping to extract inputs for the prompt template,
    #  - then the prompt template itself,
    #  - then the LLM,
    #  - then the ReAct output parser to convert LLM output into AgentAction/AgentFinish.
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_string(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # Invoke the agent once to get the first decision (action or final answer).
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    # At this point agent_step is either an AgentAction requiring a tool call,
    # or an AgentFinish which contains the final answer.

    # If the agent decided to call a tool, handle that tool invocation:
    if isinstance(agent_step, AgentAction):
        # Extract the tool name selected by the agent.
        tool_name = agent_step.tool
        # Extract the tool input the agent provided.
        tool_input = agent_step.tool_input
        # Resolve the tool object by name from our tools list.
        tool_to_call = find_tool_by_name(tools, tool_name)
        # Call the tool function with the provided input and capture the observation.
        observation = tool_to_call.func(tool_input)
        # Print the observation for visibility.
        print(f"Observation from tool: {observation}")
        # Append the action and observation so the next LLM call has context.
        intermediate_steps.append([agent_step, str(observation)])

        # Invoke the agent again now that we have the tool observation to continue reasoning.
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of 'DOG' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        # Print the agent's subsequent output (either next action or final answer).
        # print(agent_step)

    # If the agent already finished in the first step, print the final answer.
    elif isinstance(agent_step, AgentFinish):
        print("THE END====")
        print(f"Final answer: {agent_step.return_values['output']}")
    # End of script: either printed the tool-driven observation flow or final answer.
