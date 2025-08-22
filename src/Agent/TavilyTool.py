# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from pyprojroot import here
import yaml
from dotenv import load_dotenv
import os
from typing import Annotated
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId

# load_dotenv()

# with open(here("configs\\tools_config.yml")) as cfg:
#     tools_config = yaml.load(cfg, Loader=yaml.FullLoader)


def load_tavily_search_tool(tavily_search_max_results: int):
    """
    This function initializes a Tavily search tool, which performs searches and returns results
    based on user queries. The `max_results` parameter controls how many search results are
    retrieved for each query.

    The Tavily search tool is used to search external sources if sufficient data cannot be found using the other tools.

    Args:
        tavily_search_max_results (int): The maximum number of search results to return for each query.

    Returns:
        TavilySearchResults: A configured instance of the Tavily search tool with the specified `max_results`.
    """
    return TavilySearch(max_results=tavily_search_max_results)
    
    # return Command(update={
    #     "messages": [ToolMessage("Success", tool_call_id=tool_call_id)],
    #     "reason": results
    # })

# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# tavily = load_tavily_search_tool(tools_config["tavily_configs"]["max_results"])
# print(tavily.invoke("How can I decrease the amount of non-paying loans?"))