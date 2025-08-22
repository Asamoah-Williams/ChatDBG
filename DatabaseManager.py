# initialize database
# get any database schema
# execute calls to the database 
import os
import urllib.parse
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import pandas as pd
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


class DatabaseManager:
    def __init__(self, uid, password):
        connectionString=f"mssql+pyodbc://{uid}:{password}@testpfidb.database.windows.net/DBG_DATA?driver=ODBC+Driver+18+for+SQL+Server"
        self.db_engine = create_engine(connectionString)
        self.db = SQLDatabase(self.db_engine, view_support=True, schema="dbo")
        self.dialect = "mssql"


    def table_details(self, path):
        '''
        Formats table names and descriptions for prompting

        Args:
            path: the path to a csv file where each row is a table and its decription

        Returns:
            table_details: a string with table names and descriptions
        '''
        table_description = pd.read_csv(path)
        table_details = ""

        for index, row in table_description.iterrows():
            table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

        return table_details
    
    def execute_query(self, query):
        '''
        Executes SQL query 

        Args:
            query: the SQL query

        Returns:
            results: a list of query results where each row is a tuple
        '''

        executeQueryTool = QuerySQLDatabaseTool(db=self.db)
        results = executeQueryTool.invoke(query)

        return results
    

# test 

load_dotenv()
uid = os.getenv("UID") 
password = urllib.parse.quote_plus(os.getenv("PASSWORD")) 

db = DatabaseManager(uid, password)