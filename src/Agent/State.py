from typing import List, Any, Annotated, Dict, Optional, NotRequired
from typing_extensions import TypedDict
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages


# sql_query: NotRequired[Annotated[List[Any], operator.add]]
# uid: str
# results: NotRequired[Annotated[List[Any], operator.add]]
# sql_results: NotRequired[Annotated[List[Any], operator.add]]
# visualization: NotRequired[Annotated[str, operator.add]] #the type of visualization
# answer: NotRequired[Annotated[str, operator.add]] #the llms answer

class State(AgentState):
    user_question: NotRequired[str] #user's question
    sql_query: NotRequired[str] #sql query from sql tool
    results: NotRequired[List[Any]] #the results of any of the tools???
    sql_results: NotRequired[List[Any]]
    visualization: NotRequired[str]
    answer: NotRequired[str]
    visualization_reason: NotRequired[str] #reason for a certain visualization type
    formatted_data_for_visualization: NotRequired[Dict[str, Any]] #data for visualization
    error: NotRequired[str] #any error that comes up
    answer_done: NotRequired[bool]
    messages: Annotated[list, add_messages]
   