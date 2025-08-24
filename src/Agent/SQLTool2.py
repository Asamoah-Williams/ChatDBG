from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
import os
from langgraph.prebuilt import create_react_agent
import ast
from langchain_openai import ChatOpenAI
import pandas as pd
from pydantic import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from typing import List 
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
# from load_tools_config import LoadToolsConfig
from .load_tools_config import LoadToolsConfig
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import urllib
from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated, List
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from .State import State


TOOLS_CFG = LoadToolsConfig()

class Table(BaseModel):
    """
    Represents a table in the SQL database.

    Attributes:
        name (str): The name of the table in the SQL database.
    """
    name: str = Field(description="Name of table in SQL database.")

class SQLTool:
    """
    A specialized SQL agent that interacts with the MSSQL database using a Large Language Model (LLM).

    The agent handles SQL queries by mapping user questiond to relevant SQL tables based on categories like "NPL" and "GDP". It uses an extraction chain to determine the relevant tabes based on the question and then execute the queries against the appropriate tables.

    Attributes:
        db (SQLDatabase): The SQL database agent that represents the MSSQL database.

        sql_llm (ChatOpenAI): The large language model for interpreting questions to SQL.

        table_llm (ChatOpenAI): The large language model for identifying relevant tables from queries.

    Methods:
        __init__: Initializes the agent by setting up the MSSQL database and the large language models.

        table_details: Formats table names and their descriptions for prompting.

        query_to_sql: Converts user query to SQL by creating chain of operations that maps user questions to SQL tables and executes queries.

        execute_sql: Executes SQL query using the MSSQL database agent.
    """

    def __init__(self, llm: str, llm_temperature: float) -> None:
        """
        Initializes the MSSQL database and the large language models.

        
        Args:
            llm (str): The name of the OpenAI model to use.
            llm_temperature (int): The temperature for controlling the randomness of the responses.
        """

        uid = os.getenv("SQLDB_UID") 
        password = urllib.parse.quote_plus(os.getenv("SQLDB_PASSWORD"))
        connectionString=f"mssql+pyodbc://{uid}:{password}@testpfidb.database.windows.net/DBG_DATA?driver=ODBC+Driver+18+for+SQL+Server"
        db_engine = create_engine(connectionString)
        self.db = SQLDatabase(db_engine, view_support=True, schema="dbo") 
        

        self.llm = init_chat_model(llm)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = toolkit.get_tools()

    def table_details(self, path: str) -> str:
        '''
        Formats table names and descriptions for prompting

        Args:
            path: the path to a csv file where each row is a table and its decription

        Returns:
            table_details: a string with table names and descriptions
        '''

        table_description = pd.read_csv(path)
        # table_names = set()
        table_details = ""

        for index, row in table_description.iterrows():
            table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"
            # table_names.add(row['Table'])

        # self.table_names = table_names
        self.tnames_descrip = table_details

        return table_details
        

        
@tool
# def sql_tool(user_question: str) -> None:
def sql_tool(state:State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Converts user question to SQL and queries the MSSQL Database. Input should be the question from the user."""

    agent = SQLTool(
        llm=TOOLS_CFG.sqlagent_llm,
        llm_temperature=TOOLS_CFG.sqlagent_llm_temperature
    )

    table_details = agent.table_details("C:\\Users\\user\\Documents\\NPL\\LLM-WOKG\\Notebooks\\database_table_descriptions.csv")

    system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        When users request information about multiple variables, 
        you MUST create queries that include ALL requested variables in a single query or 
        coordinated set of queries.

        CRITICAL RULES:
        - If multiple variables are mentioned (e.g., "NPL_diff and DEGU_diff"), your query must encompass getting data from BOTH variables and using merging strategies
        - Never create separate queries for each variable when they should be analyzed together
        - Use JOINs, UNIONs, or subqueries as needed to combine data from multiple sources
        - If variables are from different tables, find common keys (like DATE) to join them
        - Always cast DATE keys to DATE type first

        Example: If asked for "NPL_diff and DEGU_diff trends", create a query that shows both metrics together, not separate queries for each.

        Here are some table descriptions that could help you with selecting the correct tables based on the table names.
        TABLE DESCRIPTIONS: {table_details}

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        IF MORE THAN ONE VARIABLE IS BEING REQUESTED THE 

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        THE RESULTS SHOULD ONLY BE IN THE FOLLOWING FORMAT, SO MAKE SURE THE SQL QUERY ACCOMMODATES THIS:
        [[x, y]]
        or 
        [[label, x, y]]

        If the question does not include specific years then use ONLY THE TWO YEARS OF DATA IN DESCENDING ORDER.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect=agent.db.dialect,
            top_k=5,
            table_details = table_details
        )

    agent = create_react_agent(
        agent.llm,
        agent.tools,
        prompt=system_prompt,
    )

    user_question = state.get("user_question")
    input = {"messages": [{"role": "user", "content": user_question}]}

    ans = agent.invoke(input)
    ans_dict = dict()

    for message in ans["messages"]:
        if type(message) == ToolMessage: 
            if message.name  == 'sql_db_query':
                try:
                    data = message.content 
                    data = ast.literal_eval(data)
                    ans_dict["data"] = data
                except:
                    print("DATA", data)
                    ans_dict["data"] = []
        elif type(message) == AIMessage:
            if message.tool_calls and message.tool_calls[0].get("name") == 'sql_db_query':
                    query = message.tool_calls[0].get("args")["query"]
                    ans_dict["sql_query"] = query

    try:
        print("we entered here")
        return Command(update={
            "sql_results": ans_dict["data"],
            "sql_query": ans_dict["sql_query"],
            "messages": [ToolMessage("SQL tool successful", tool_call_id=tool_call_id)]
        })
    except:
        print("the error is happening here")


    
# sql = SQLTool(llm="gpt-3.5-turbo", llm_temperature= 0.0)
# ans = sql.question_to_sql("When the community bank leverage ratio fell, how did NPL change in the next quarter?")
# print (ans)
# sql.table_info()

# sql_tool("i want a line graph showing the exchange and NPL within 2022 and 2025")