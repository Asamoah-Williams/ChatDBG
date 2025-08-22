from langchain_chroma import Chroma
import os
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import OpenAIEmbeddings
from src.utils.load_tools_config import LoadToolsConfig
from typing import Annotated
from langgraph.prebuilt import InjectedState
from src.Agent.State import State
from langgraph.types import Command
from langchain_core.messages import ToolMessage


TOOLS_CFG = LoadToolsConfig()


class GuideRAGTool:
    """
    A tool for retrieving relevant instructions in the Key Risk Indicator Guide using a Retrieval-Augmented Generation (RAG) approach with vector embeddings.

    The tool leverages a pre-trained OpenAI embedding. It then uses these embeddings to query a cloud-based Chroma vector database to retrieve the top-k rekevant guide points from a specific collection stored in the database.

    Attributes:
        embedding_model (str): The name of the OpenAI embedding model used for generating vector representations of queries.
        vector_store (Chroma): The Chroma vector database connected to the specified collection and embedding model.
        k (int): The top-k nearest neighbor guide instructions to retrieve from the database.

    Methods:
        __init__: Initializes the tool with the specified embedding model, vector database, and retrieval parameters.

    """

    def __init__(self,embedding_model: str, collection_name: str, k: int) -> None:
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
        self.k = k

@tool
def lookup_kri_guide(question: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search through the key risk indicator guide and find the answer to the query. Input should be the user's question."""
    rag_tool = GuideRAGTool(
        embedding_model=TOOLS_CFG.guiderag_embedding_model,
        collection_name=TOOLS_CFG.guiderag_collection_name,
        k = TOOLS_CFG.guiderag_k
    )
    docs = rag_tool.vector_store.similarity_search(query=question, k=rag_tool.k)
    retrieved_content = "\n\n".join([ans.page_content for ans in docs])
    
    # return retrieved_content
    return Command(update={
        "messages": [ToolMessage(f"The answer based on the tool is {retrieved_content}", tool_call_id=tool_call_id)],
        "results": [retrieved_content]
    })


# cont = lookup_kri_guide("what are the rules for the kri_name column")
# print(cont)