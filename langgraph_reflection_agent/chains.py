# langgraph_reflection_agent/chains.py
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI agent that grades a tweets. You will be given a tweet and you need to reflect on it and provide feedback on how to improve it.
            Always provide detailed, specific and constructive feedback: including requests for length, tone, clarity, grammar, and engagement.""",
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
# Reflection chain
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
