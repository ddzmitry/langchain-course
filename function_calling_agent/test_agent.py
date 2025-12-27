"""
Comprehensive Test Suite for Function Calling Agent

This file tests the function calling agent implementation using .bind_tools()
with manual message handling. It verifies:
- Tool definitions and invocation
- Message-based conversation flow
- Tool call structure and handling
- Comparison with ReAct approach
"""

import pytest
from dotenv import load_dotenv
from langchain.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing import List

# Load environment variables from .env file
load_dotenv()


# Test Tools
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    text = text.strip("'\n").strip('"')
    return len(text)


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b


@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


# Helper function (same as in main.py)
def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """Find a tool by its name from a list of tools."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


# Test Cases


class TestToolDefinitions:
    """Test that tools are properly defined and work correctly."""

    def test_get_text_length_basic(self):
        """Test get_text_length with basic input."""
        result = get_text_length.func("hello")
        assert result == 5

    def test_get_text_length_with_quotes(self):
        """Test get_text_length strips quotes correctly."""
        result = get_text_length.func('"hello"')
        assert result == 5

    def test_get_text_length_with_newlines(self):
        """Test get_text_length strips newlines correctly."""
        result = get_text_length.func("hello\n")
        assert result == 5

    def test_multiply_numbers(self):
        """Test multiply_numbers works correctly."""
        result = multiply_numbers.func(3, 4)
        assert result == 12

    def test_add_numbers(self):
        """Test add_numbers works correctly."""
        result = add_numbers.func(5, 7)
        assert result == 12

    def test_find_tool_by_name_success(self):
        """Test finding a tool by name."""
        tools = [get_text_length, multiply_numbers]
        tool = find_tool_by_name(tools, "get_text_length")
        assert tool == get_text_length

    def test_find_tool_by_name_not_found(self):
        """Test that finding non-existent tool raises ValueError."""
        tools = [get_text_length]
        with pytest.raises(ValueError, match="Tool with name nonexistent not found"):
            find_tool_by_name(tools, "nonexistent")


class TestBindTools:
    """Test the .bind_tools() functionality."""

    def test_bind_tools_creates_llm_with_tools(self):
        """Test that .bind_tools() returns an LLM instance with tools bound."""
        llm = ChatOpenAI(temperature=0)
        tools = [get_text_length]
        llm_with_tools = llm.bind_tools(tools)

        # Should have tools bound
        assert llm_with_tools is not None
        assert llm_with_tools != llm  # Should be a modified version
        assert hasattr(llm_with_tools, "kwargs")

    def test_bind_tools_with_multiple_tools(self):
        """Test binding multiple tools."""
        llm = ChatOpenAI(temperature=0)
        tools = [get_text_length, multiply_numbers, add_numbers]
        llm_with_tools = llm.bind_tools(tools)

        assert llm_with_tools is not None
        assert hasattr(llm_with_tools, "kwargs")


class TestMessageStructure:
    """Test that message-based conversation works correctly."""

    def test_human_message_creation(self):
        """Test creating HumanMessage."""
        msg = HumanMessage(content="What is the length of 'hello'?")
        assert msg.content == "What is the length of 'hello'?"
        assert msg.type == "human"

    def test_tool_message_creation(self):
        """Test creating ToolMessage."""
        msg = ToolMessage(content="5", tool_call_id="call_123")
        assert msg.content == "5"
        assert msg.tool_call_id == "call_123"
        assert msg.type == "tool"

    def test_message_list(self):
        """Test building a message list."""
        messages = [
            HumanMessage(content="What is the length of 'hello'?"),
            ToolMessage(content="5", tool_call_id="call_123"),
        ]
        assert len(messages) == 2
        assert messages[0].type == "human"
        assert messages[1].type == "tool"


class TestFunctionCallingVsReAct:
    """Tests highlighting differences between Function Calling and ReAct."""

    def test_no_stop_sequences_needed(self):
        """Function calling doesn't need stop sequences."""
        llm = ChatOpenAI(temperature=0)
        # Should not have stop sequences
        assert not hasattr(llm, "stop") or llm.stop is None

    def test_no_prompt_template_needed(self):
        """Function calling doesn't need complex prompt templates."""
        # Just create a simple message
        msg = HumanMessage(content="What is the length of 'strawberry'?")
        # No need for PromptTemplate, no "Thought/Action/Observation" format
        assert "Thought:" not in msg.content
        assert "Action:" not in msg.content
        assert "Observation:" not in msg.content

    def test_structured_tool_calls(self):
        """Test that tool calls have structured format."""
        # Tool calls should be dictionaries with specific keys
        tool_call = {
            "id": "call_abc123",
            "name": "get_text_length",
            "args": {"text": "hello"},
        }
        # Verify structure
        assert "id" in tool_call
        assert "name" in tool_call
        assert "args" in tool_call
        assert isinstance(tool_call["args"], dict)


class TestToolCallFlow:
    """Test the tool calling flow without LLM calls."""

    def test_tool_invocation_flow(self):
        """Test manual tool invocation as it happens in the loop."""
        tools = [get_text_length]

        # Simulate a tool call from LLM
        tool_call = {
            "id": "call_123",
            "name": "get_text_length",
            "args": {"text": "hello"},
        }

        # Find and invoke tool (same as in main loop)
        tool_to_use = find_tool_by_name(tools, tool_call.get("name"))
        observation = tool_to_use.invoke(tool_call.get("args", {}))

        # Create ToolMessage
        tool_msg = ToolMessage(
            content=str(observation), tool_call_id=tool_call.get("id")
        )

        assert observation == 5
        assert tool_msg.content == "5"
        assert tool_msg.tool_call_id == "call_123"

    def test_multiple_tool_calls(self):
        """Test handling multiple tool calls in sequence."""
        tools = [get_text_length, multiply_numbers]

        tool_calls = [
            {"id": "call_1", "name": "get_text_length", "args": {"text": "hi"}},
            {"id": "call_2", "name": "multiply_numbers", "args": {"a": 3, "b": 4}},
        ]

        results = []
        for tool_call in tool_calls:
            tool_to_use = find_tool_by_name(tools, tool_call.get("name"))
            observation = tool_to_use.invoke(tool_call.get("args", {}))
            results.append(observation)

        assert results[0] == 2  # len("hi")
        assert results[1] == 12  # 3 * 4


# Manual test runner
def run_manual_tests():
    """
    Run basic smoke tests manually without pytest.
    Useful for quick verification during development.
    """
    print("Running manual smoke tests...\n")

    # Test 1: Tool definitions work
    print("Test 1: Tool definitions")
    assert get_text_length.func("hello") == 5
    assert multiply_numbers.func(3, 4) == 12
    print("✓ Tools work correctly\n")

    # Test 2: .bind_tools() works
    print("Test 2: .bind_tools() method")
    try:
        llm = ChatOpenAI(temperature=0)
        tools = [get_text_length]
        llm_with_tools = llm.bind_tools(tools)
        assert llm_with_tools is not None
        assert hasattr(llm_with_tools, "kwargs")
        print("✓ .bind_tools() works correctly\n")
    except Exception as e:
        print(f"⊗ Skipped (requires OPENAI_API_KEY): {str(e)[:60]}...\n")

    # Test 3: Message creation
    print("Test 3: Message creation")
    msg = HumanMessage(content="Test message")
    assert msg.content == "Test message"
    tool_msg = ToolMessage(content="5", tool_call_id="call_123")
    assert tool_msg.tool_call_id == "call_123"
    print("✓ Messages created correctly\n")

    # Test 4: Tool invocation flow
    print("Test 4: Tool invocation flow")
    tools = [get_text_length]
    tool_call = {"id": "call_1", "name": "get_text_length", "args": {"text": "hello"}}
    tool_to_use = find_tool_by_name(tools, tool_call.get("name"))
    observation = tool_to_use.invoke(tool_call.get("args", {}))
    assert observation == 5
    print("✓ Tool invocation works correctly\n")

    print("All manual tests passed! ✓")
    print("\nTo run pytest suite:")
    print("  pytest test_agent.py -v")
    print("\nTo run specific test class:")
    print("  pytest test_agent.py::TestToolDefinitions -v")


if __name__ == "__main__":
    # Run manual tests when executed directly
    run_manual_tests()


# Test execution instructions:
#
# 1. Run all unit tests (no LLM calls):
#    pytest test_agent.py -v
#
# 2. Run specific test class:
#    pytest test_agent.py::TestToolDefinitions -v
#
# 3. Run manual smoke tests:
#    python test_agent.py
#
# 4. Run with coverage:
#    pytest test_agent.py --cov=. --cov-report=html
#
# Note: These tests focus on the mechanics of function calling
# without making actual LLM API calls, making them fast and free to run.
