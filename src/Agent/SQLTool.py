from sqlalchemy import create_engine
# from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
import os
import ast
from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from typing import List 
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from DatabaseManager import DatabaseManager
from src.utils.load_tools_config import LoadToolsConfig
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import urllib
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated, List
# from langgraph.prebuilt import InjectedState
# from src.Agent.State import State
from langgraph.types import Command
from langchain_core.messages import ToolMessage

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

        self.sql_llm = ChatOpenAI(model=llm, temperature=llm_temperature) 
        self.table_llm = ChatOpenAI(model=llm, temperature=llm_temperature)
        self.banned_tables = ["transformed_GDP"]
        self.table_names = set()
        self.tnames_descrip = ""


    def table_details(self, path: str) -> str:
        '''
        Formats table names and descriptions for prompting

        Args:
            path: the path to a csv file where each row is a table and its decription

        Returns:
            table_details: a string with table names and descriptions
        '''

        table_description = pd.read_csv(path)
        table_names = set()
        table_details = ""

        for index, row in table_description.iterrows():
            table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"
            table_names.add(row['Table'])

        self.table_names = table_names
        self.tnames_descrip = table_details

        return table_details, table_names
    

    def question_to_sql(self, query: str) -> str:
        """
        Converts user query to SQL by creating chain of operations that maps user questions to SQL tables and executes queries.

        Args:
            query: User's query

        Returns:
            sql_query: Valid SQL query that represents user's query 
        """

        table_details, table_names = self.table_details("Notebooks/database_table_descriptions.csv")
        table_names = str(list(table_names))
        banned_tables = str(self.banned_tables)

        # print(type(table_details), type(table_names), type(banned_tables))
        # print("table deatils", table_details)
        # print("table names", table_names)
        # print("banned tables", banned_tables)
        

        table_details_prompt = f"""Return the names of ALL the SQL tables that ARE RELEVANT to the user question.
        Choose ONLY from this list (do not invent or hallucinate): {table_names}

        NEVER USE: {banned_tables}    

        {table_details}    
        
        If a concept lacks its transformed table, use its raw table.
        Cross-check with the table and descriptions below, omit if unsure of table name.
        Tables and Descriptions: 
        Always favor transformed data over raw data unless the user specifies that they want raw data.
        

        Here are some examples of WRONG TABLES GIVEN A USER QUESTION:
        1. User question: What is the correlation between GDP and NPL?
        Wrong Tables: transformed_GDP, transformed_NPL THIS IS WRONG

        2. User question: what is the difference between the NPL and the exchange rate?
        Wrong Tables: NPL_Raw, DEGU_Raw THIS IS WRONG

        3. Plot exchange rate against community GDP
        Wrong Tables: DEGU_Raw, transformed_GDP THIS IS WRONG
        

        Here are some examples of CORRECT TABLES GIVEN A USER QUESTION:
        1. User question: What is the correlation between GDP and NPL?
        Correct Tables: GDP_Raw, transformed_NPL

        2. User question: what is the difference between the raw NPL and the transformed NPL?
        Correct Tables: NPL_Raw, transformed_NPL

        3. Plot exchange rate against community bank leverage ratio
        Correct Tables: transformed_DEGU, transformed_CBLR

        ALWAYS CROSS-CHECK TO MAKE SURE THAT THE RELEVANT TABLE NAMES YOU HAVE GENERATED ARE PART OF {table_names}.
        If the tables aren't do not return them as part of the answer.
        """

        def get_tables(tables: List[Table]) -> List[str]:
            # tables  = [table.name if table.name in self.table_names else  for table in tables]
            final_tables = []

            for table in tables:
                # print("\n NAME", table.name)
                # ill have to do the alias thing here as well i guess
                if table.name in self.table_names:
                    final_tables.append(table.name)

            # print(final_tables)

            return final_tables
        

        # this SQL query chain must have a prompt with the [[x, y]] and [[label, x, y]] thing 
        # i also need to validate the sql query to make sure everything is fine
        query_chain = create_sql_query_chain(self.sql_llm, self.db)
        query_chain.get_prompts()[0].pretty_print()

        # convert question to table needed
        table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, self.table_llm, system_message=table_details_prompt) | get_tables


        # convert question to sql query
        generate_query = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

        sql_query = generate_query.invoke(
                {"question": query}
                )
        
        return sql_query
    

    def execute_sql(self, sql_query):
        '''
        Executes SQL query 

        Args:
            query: the SQL query

        Returns:
            results: a list of query results where each row is a tuple
        '''

        executeQueryTool = QuerySQLDatabaseTool(db=self.db)
        results = executeQueryTool.invoke(sql_query)

        return results


        
@tool
# def sql_tool(state: Annotated[State, InjectedState]) -> str:
def sql_tool(user_question: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    # i can put all of this in a try and except block like multiple nested ones?
    """Converts user question to SQL and queries the MSSQL Database. Input should be the question from the user."""
    # Create an instance of MSSQLAgent
    agent = SQLTool(
        llm=TOOLS_CFG.sqlagent_llm,
        llm_temperature=TOOLS_CFG.sqlagent_llm_temperature
    )

    sql_query = agent.question_to_sql(user_question)
    data = agent.execute_sql(sql_query)
     #error when i run certain prompts, what is the correlation between GDP and NPL

    # print("from sql tool", data)

    try:
        data = ast.literal_eval(data)
        return Command(update={
            "messages": [ToolMessage(f"Success generating data from sql. SQL QUERY is {sql_query}. Data from sql query is {data}", tool_call_id=tool_call_id)],
            "results": [data],
            "sql_query": [sql_query]
        })
    except:
        return Command(update={
            "results": [[]],
            "sql_query": None,
            "messages": [ToolMessage("Query returned no results", tool_call_id=tool_call_id)]
        })

    
    # if not data or data.strip() == "":
    #     return Command(update={
    #         "results": [],
    #         "sql_query": None,
    #         "messages": [ToolMessage("Query returned no results", tool_call_id=tool_call_id)]
    #     })

    

    # return Command(update={
    #     "messages": [ToolMessage(f"Success generating data from sql. SQL QUERY is {sql_query}. Data from sql query is {data}", tool_call_id=tool_call_id)],
    #     "results": data,
    #     "sql_query": sql_query
    # })


    
# sql = SQLTool(llm="gpt-3.5-turbo", llm_temperature= 0.0)
# ans = sql.question_to_sql("i want the gdp for 2025")
# print (ans)