# Custom Callback Handler for Function Calling Agent
# This callback handler provides detailed logging during agent execution
# Unlike ReAct agents, function calling agents work with structured tool calls

from typing import Any, Dict, List
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for monitoring function calling agent execution.

    This handler intercepts key events during agent execution:
    - When the LLM is called (on_llm_start)
    - When the LLM responds (on_llm_end)
    - When a tool is called (on_tool_start)
    - When a tool finishes (on_tool_end)

    For function calling agents, the LLM responses include:
    - Regular text messages
    - Structured tool calls with function names and arguments
    - Tool responses containing execution results
    """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """
        Called when the LLM is about to be invoked.

        For function calling agents, this shows the complete conversation history
        including previous tool calls and results.

        Args:
            serialized: Serialized representation of the LLM
            prompts: List of prompt strings sent to the LLM
            **kwargs: Additional keyword arguments (may include 'messages' for chat models)
        """
        print("\n" + "="*70)
        print("LLM INVOCATION START")
        print("="*70)

        # Check if this is a chat model with messages (function calling uses this)
        if "messages" in kwargs:
            print("\nMessages sent to LLM:")
            for i, messages in enumerate(kwargs["messages"]):
                print(f"\nMessage batch {i + 1}:")
                for msg in messages:
                    print(f"  - {msg}")
        else:
            # Fallback for non-chat models
            print(f"\nPrompt to LLM:")
            for i, prompt in enumerate(prompts):
                print(f"\nPrompt {i + 1}:")
                print(prompt)

        print("="*70 + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        Called when the LLM has finished generating a response.

        For function calling agents, the response may contain:
        - A text message (final answer)
        - Tool calls with structured arguments
        - Both text and tool calls

        Args:
            response: The LLMResult containing generations and metadata
            **kwargs: Additional keyword arguments
        """
        print("\n" + "="*70)
        print("LLM RESPONSE")
        print("="*70)

        # Iterate through all generations (usually just one)
        for i, generation in enumerate(response.generations):
            print(f"\nGeneration {i + 1}:")

            for j, gen in enumerate(generation):
                print(f"  Response {j + 1}:")

                # Check if this is a chat message with potential tool calls
                if hasattr(gen, "message"):
                    msg = gen.message
                    print(f"    Content: {msg.content}")

                    # Function calling agents include tool_calls in the message
                    if hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
                        tool_calls = msg.additional_kwargs["tool_calls"]
                        print(f"    Tool Calls: {len(tool_calls)}")
                        for tc in tool_calls:
                            print(f"      - Function: {tc['function']['name']}")
                            print(f"        Arguments: {tc['function']['arguments']}")
                else:
                    # Fallback for simple text responses
                    print(f"    Text: {gen.text}")

        print("="*70 + "\n")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """
        Called when a tool is about to be executed.

        This is triggered after the LLM returns a tool call and the AgentExecutor
        begins executing the requested tool.

        Args:
            serialized: Serialized representation of the tool
            input_str: The input string passed to the tool
            **kwargs: Additional keyword arguments
        """
        tool_name = serialized.get("name", "Unknown Tool")
        print(f"\n{'~'*70}")
        print(f"TOOL EXECUTION START: {tool_name}")
        print(f"{'~'*70}")
        print(f"Input: {input_str}")
        print(f"{'~'*70}\n")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """
        Called when a tool has finished execution.

        The output will be sent back to the LLM as context for the next iteration.

        Args:
            output: The string output returned by the tool
            **kwargs: Additional keyword arguments
        """
        print(f"\n{'~'*70}")
        print("TOOL EXECUTION END")
        print(f"{'~'*70}")
        print(f"Output: {output}")
        print(f"{'~'*70}\n")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        """
        Called when a tool execution raises an error.

        The AgentExecutor can handle this gracefully and retry or inform the LLM.

        Args:
            error: The exception that was raised
            **kwargs: Additional keyword arguments
        """
        print(f"\n{'!'*70}")
        print("TOOL EXECUTION ERROR")
        print(f"{'!'*70}")
        print(f"Error: {str(error)}")
        print(f"{'!'*70}\n")


# Key Differences from ReAct Callback Handling:
#
# 1. MESSAGE STRUCTURE:
#    - ReAct: Simple text with "Thought:", "Action:", "Action Input:" parsing
#    - Function Calling: Structured messages with tool_calls in additional_kwargs
#
# 2. TOOL CALLS:
#    - ReAct: Extracted from parsed text (fragile)
#    - Function Calling: Structured JSON in message.additional_kwargs["tool_calls"]
#
# 3. CONVERSATION FLOW:
#    - ReAct: Linear text with stop sequences
#    - Function Calling: Alternating messages (assistant -> tool -> assistant -> ...)
#
# 4. ERROR HANDLING:
#    - ReAct: Parsing errors common if format is wrong
#    - Function Calling: Schema validation ensures correct format
#
# This callback handler is designed to work with both patterns but provides
# more detailed information for function calling agents due to the structured data.