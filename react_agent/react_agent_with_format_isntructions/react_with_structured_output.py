# This block provides inline documentation for the rest of the module.
# It is placed at the end of the file to avoid modifying existing logic,
# and to serve as a single consolidated comment region describing each part.

# Top-level environment loading:
# The dotenv loader reads environment variables from a .env file so that
# credentials (e.g., OpenAI API keys) and other configuration are available.

# Imports and high-level libraries:
# - langchain tools and agent utilities are used to build the agent and glue
#   the language model to external tools.
# - ChatOpenAI is the LLM wrapper used to call GPT-4.
# - TavilySearch is a search tool used by the agent to fetch web results.
# - PydanticOutputParser and PromptTemplate are used to ensure the agent
#   returns structured outputs conforming to the AgentResponse schema.
# - RunnableLambda is used to transform the agent's result in a pipeline style.

# Tools list:
# The `tools` variable is a list of tool instances the agent can call.
# Currently it contains a single TavilySearch instance for searching web data.

# Language model:
# `llm` is instantiated as a ChatOpenAI model configured for "gpt-4".
# This object is responsible for generating the agent's natural language output.

# Prompt and output parsing:
# - `react_prompt` holds the base prompt template string (imported from prompt).
# - `output_parser` wraps the AgentResponse pydantic model to produce format
#   instructions and to parse the final LLM output into a typed object.
# - `react_prompt_with_format_instructions` is a PromptTemplate that injects
#   the parser's format instructions into the prompt so the model knows how
#   to structure its reply.

# Agent construction:
# `create_react_agent` builds a REACT-style agent (Reasoning + Acting) using the
# provided llm, tools, and prompt. This is the core decision-making component.

# Agent executor:
# `AgentExecutor` wraps the agent and manages tool execution, state, and
# orchestration. `verbose=True` enables detailed logging of the agent's steps.

# Chain alias:
# `chain` points to the agent_executor for a simpler, chain-like invocation API.

# main() behavior:
# - The main function invokes the chain with an input dictionary containing the
#   user's question. The input asks the agent to search for restaurants and to
#   "show the thinking process" (the chain may include the agent's scratchpad).
# - The returned value is passed through a RunnableLambda which extracts the
#   "output" field from the agent's response mapping. This isolates the final
#   human-readable output string.
# - The result is printed to stdout so you can see the agent's answer when
#   running the script.

# Notes and tips:
# - If you want the agent to return a parsed Pydantic object rather than a
#   raw string, call the output_parser on the agent response instead of
#   extracting "output" as plain text.
# - To add more capabilities, append tool instances to the `tools` list and
#   ensure the prompt's tool descriptions are updated accordingly.
# - For debugging, keep `verbose=True` on AgentExecutor; to reduce console noise,
#   set it to False.
# - Always secure API keys in environment variables; avoid hard-coding secrets.

# End of comments for this module.
from dotenv import load_dotenv

load_dotenv()
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

# Define the output parser using the Pydantic model

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
react_prompt_with_format_instructions = PromptTemplate(
    template=react_prompt,
    input_variables=[
        "input",
        "tools",
        "tool_names",
        "format_instructions",
        "agent_scratchpad",
    ],
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(
    lambda x: x["output"]
)  # Extract the output string from the result dictionary


chain = (
    agent_executor | extract_output | structured_llm
)  # Parse the output into a Pydantic object will be called like a tool


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 great places to eat in Charlotte , NC Southend/LoSo area, list the details, make sure you show the thinking process"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
