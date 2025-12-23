from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Schema for a source used by the agent
    """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for agent response
    with answer and sources
    """

    answer: str = Field(description="The answer for the question")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer",
    )


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from react-from-scratch")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Find me 3 best places to hike in Charlotte NC, provide sources with valid links"
                ),
            ]
        }
    )
    print(result)


if __name__:
    main()
