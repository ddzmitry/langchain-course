from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


# create a tool function that will take number and tripple it add comements for AI agent
@tool
def triple_number(number: float) -> float:
    """Triples the given float number."""
    return float(number) * 3


@tool
def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """Converts Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


tools = [TavilySearch(max_results=3), triple_number, convert_celsius_to_fahrenheit]
# Will send tools to LLM on every request
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
