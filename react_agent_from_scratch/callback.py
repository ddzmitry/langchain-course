from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Tuple


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for agent execution."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Print out when the LLM starts."""
        print(f"***Prompt to LLM was:*** \n {prompts[0]}\n")
        print("***********")
        for prompt in prompts:
            print(prompt)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Print out when the LLM ends."""
        print(f"***LLM response was:*** \n {response.generations[0][0].text}\n")
        print("***********")
