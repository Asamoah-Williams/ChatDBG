from ..utils.load_tools_config import LoadToolsConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from .State import State
# from SQLTool import sql_tool
from .TavilyTool import load_tavily_search_tool
from .SQLTool2 import sql_tool
from .RAGTool import lookup_kri_guide
from .VisualizationTool import visualization_tool
from .tool_node import route_tools
from langgraph.prebuilt import ToolNode
from .final_node import finalizer
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg2
from psycopg import Connection
from langchain_core.messages import trim_messages

TOOLS_CFG = LoadToolsConfig()

def build_graph():
    llm = ChatOpenAI(model=TOOLS_CFG.primary_agent_llm, 
                     temperature=TOOLS_CFG.primary_agent_llm_temperature)
    
    DB_CONFIG = {
    "host": "localhost",
    "database": "memory_llm",
    "user": "dbg_llm", 
    "password": "dbgllm",
    "port": 5432
    }
    
    con = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    conn = Connection.connect(con, **connection_kwargs)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    graph_builder = StateGraph(State)

    search_tool = load_tavily_search_tool(tavily_search_max_results=TOOLS_CFG.tavily_search_max_results)
    tools = [
        search_tool,
        sql_tool,
        lookup_kri_guide,
        visualization_tool
    ]

    primary_llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools=[
        search_tool,
        sql_tool,
        lookup_kri_guide,
        visualization_tool
        ]  )

    def chatbot(state: State):
        """Executes the primary language model with tools bound and returns the generated message."""
        messages = state["messages"]

        trimmed_messages = trim_messages(
            messages,
            max_tokens=4000,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True,
            # allow_partial=True
        )

        return {"messages": [primary_llm_with_tools.invoke(trimmed_messages)]}
    

    def capture(state: State):
        """Executes the primary language model with tools bound and returns the generated message."""
        messages = state.get("messages")

        # reset all the others here 

        return {
            "user_question": messages[-1].content,
            "sql_query": "",
            "sql_results": [],
            "answer": "",
        }



    # adding nodes
    graph_builder.add_node("capture", capture)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("finalizer", finalizer)

    # adding edges 
    graph_builder.add_edge(START, "capture")
    graph_builder.add_edge("capture", "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "finalizer"},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("finalizer", END)
    

    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph




# graph = build_graph()
# # display(Image(graph.get_graph().draw_mermaid_png(output_file_path="graph.png")))
# input_message = {"role": "user", "content": "according to the kri guide what are the rules for adding the threshold limits column to the risk report?"}

# WATCH_KEYS = {"visualization", "formatted_data_for_visualization", "answer", "answer_done"}
# events = graph.stream(
#             {"messages": [input_message]}, stream_mode="values"
#         )

# for event in events:
#     payload = {k: v for k, v in event.items() if k in WATCH_KEYS}
#     if payload:
#         print(payload)

