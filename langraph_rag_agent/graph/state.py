from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Representation of graph state

    Attributes:
        question: question
        generation: LLM generation
        web_search: wether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
