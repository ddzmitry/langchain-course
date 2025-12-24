from typing import Any, Dict, List, Optional

from pydantic import AnyUrl, BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The answer for the question")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer",
    )
