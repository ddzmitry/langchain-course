# langgraph_reflection_agent/chains.py
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ReflectionOutput(BaseModel):
    """Structured output for reflection with feedback and score."""

    feedback: str = Field(description="Detailed feedback on how to improve the tweet")
    score: float = Field(
        description="Score from 1 to 10 rating the quality of the tweet (10 is best)",
        ge=1,
        le=10,
    )
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI agent that grades tweets. You will be given a tweet and you need to reflect on it and provide feedback on how to improve it.
            Always provide detailed, specific and constructive feedback: including requests for length, tone, clarity, grammar, and engagement.
            Your feedback will be used by another AI agent to improve the tweet. Provide a score from 1 to 10 (10 is best) to rate the tweet quality.
            """,
        ),
        # Place to insert the conversation messages, plug messages into the prompt history
        MessagesPlaceholder(variable_name="messages"),
        # User prompt to reflect on the tweet
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a twitter techi influencer tasked with writing excellent twitter posts.
            Generate the best twitter post possible for the users request.
            If the user provides critique, respond with a revised version of your previous attempts.""",
        ),
        # Place to insert the conversation messages, plug messages into the prompt history
        MessagesPlaceholder(variable_name="messages"),
        # User prompt to generate improved tweet
    ]
)
# LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# LLM with structured output for reflection
llm_with_structure = llm.with_structured_output(ReflectionOutput)
# Reflection chain
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm_with_structure
