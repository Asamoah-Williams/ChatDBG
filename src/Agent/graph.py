from ..utils import LoadToolsConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from State import State
from SQLTool import sql_tool
from TavilyTool import load_tavily_search_tool
from RAGTool import lookup_kri_guide
from VisualizationTool import visualization_tool
from tool_node import route_tools
from langgraph.prebuilt import ToolNode
from final_node import finalizer
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
    # checkpointer = PostgresSaver.from_conn_string(conn_string=con)
    # checkpointer = PostgresSaver(con)

    # checkpointer.setup()
    
    # summarization_node = SummarizationNode( 
    #     token_counter=count_tokens_approximately,
    #     # model=model,
    #     max_tokens=384,
    #     max_summary_tokens=128,
    #     output_messages_key="messages",
    # )

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

        # reset answer and answer done everytime this starts 

        trimmed_messages = trim_messages(
            messages,
            max_tokens=4000,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True,
            # allow_partial=True
        )

        return {"messages": [primary_llm_with_tools.invoke(trimmed_messages)]}
    




    # adding nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("finalizer", finalizer)

    # adding edges 
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", "__end__": "finalizer"},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    # graph_builder.add_edge("chatbot", "finalizer")
    graph_builder.add_edge("finalizer", END)
    
    # memory = MemorySaver()
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

