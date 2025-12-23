from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that search over the internet

    Args:
        query (str): The query to search

    Returns:
        str: The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query, max_results=1)


@tool
def add_some_snow(query: str) -> str:
    """This tool adds some snow emojies if today is a winter time

    Args:
        query (str): query to add snow

    Returns:
        str: string with snowflakes
    """
    print("=====It is cold======")
    return f"❄️❄️❄️ {query} ❄️❄️❄️"


@tool
def get_time() -> str:
    """
    This tool returns time
    """
    print("=====It is cold======")
    return f"Today is {datetime.now()}"


@tool
def add_some_sun(query: str) -> str:
    """This tool adds some snow emojies if today is a summer time

    Args:
        query (str): query to add snow

    Returns:
        str: string with snowflakes
    """
    print("=====It is sunny and suimmer======")
    return f"☀️☀️☀️ {query} ☀️☀️☀️"


llm = ChatOpenAI()
# model="gpt-5"
tools = [search, add_some_snow, add_some_sun, get_time]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from react-from-scratch")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Dzmitry"),
                AIMessage(content="Hello Dzmitry"),
                HumanMessage(
                    content="What is weather in Tokyo, find time and add season with the messages"
                ),
            ]
        }
    )
    # print(result)
    for i in result.get("messages"):
        print(i)


if __name__:
    main()
