from dotenv import load_dotenv

load_dotenv()
from langchain.tools import tool
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Define the output parser using the Pydantic model

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=react_prompt,
    input_variables=[
        "input",
        "tools",
        "tool_names",
        "format_instructions",
        "agent_scratchpad",
    ],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 great places to eat in Charlotte , NC Southend/LoSo area, list the details, make sure you show the thinking process"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
