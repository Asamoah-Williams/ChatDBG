from typing import List, Any, Annotated, Dict, Optional, NotRequired
from typing_extensions import TypedDict
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages

class State(AgentState):
    question: NotRequired[str] #user's question
    # uid: str
    # sql_query: NotRequired[str] #sql query from sql tool
    sql_query: NotRequired[Annotated[List[Any], operator.add]]
    # results: NotRequired[List[Any]] #the results of any of the tools???
    results: NotRequired[Annotated[List[Any], operator.add]]
    # visualization: Annotated[str, operator.add]
    visualization: NotRequired[Annotated[str, operator.add]] #the type of visualization
    # answer: NotRequired[Annotated[str, operator.add]] #the llms answer
    answer: NotRequired[str]
    visualization_reason: NotRequired[Annotated[str, operator.add]] #reason for a certain visualization type
    formatted_data_for_visualization: NotRequired[Dict[str, Any]] #data for visualization
    error: NotRequired[str] #any error that comes up
    answer_done: NotRequired[bool]
    messages: Annotated[list, add_messages]
   