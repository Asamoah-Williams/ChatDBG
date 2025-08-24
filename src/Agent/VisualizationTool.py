from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import ast
from dotenv import load_dotenv
import os
from langchain_core.tools import tool, InjectedToolCallId
from src.utils.load_tools_config import LoadToolsConfig
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, List
from src.Agent.State import State
from .State import State
from langgraph.prebuilt import InjectedState
from src.utils.load_project_configs import LoadProjectConfigs
# from load_project_configs import LoadProjectConfigs
from ..utils.graph_instructions import graph_instructions
import json
import datetime

# load_dotenv()
TOOLS_CFG = LoadToolsConfig()
PROJECTS_CFG = LoadProjectConfigs()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")

class VisualizationTool:
    def __init__(self, llm, llm_temperature):
        self.llm = ChatOpenAI(model=llm, temperature=llm_temperature)
        # self.state = state

    def visualization_type(self, question, query, results):
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
                You are an AI assistant that recommends appropriate data visualizations. Based on the user's question, SQL query, and query results, suggest the most suitable type of graph or chart to visualize the data. If no visualization is appropriate, indicate that.

                Available chart types and their use cases:
                - Bar Graphs: Best for comparing categorical data or showing changes over time when categories are discrete and the number of categories is more than 2. Use for questions like "What are the sales figures for each product?" or "How does the population of cities compare? or "What percentage of each city is male?"
                - Horizontal Bar Graphs: Best for comparing categorical data or showing changes over time when the number of categories is small or the disparity between categories is large. Use for questions like "Show the revenue of A and B?" or "How does the population of 2 cities compare?" or "How many men and women got promoted?" or "What percentage of men and what percentage of women got promoted?" when the disparity between categories is large.
                - Scatter Plots: Useful for identifying relationships or correlations between two numerical variables or plotting distributions of data. Best used when both x axis and y axis are continuous. Use for questions like "Plot a distribution of the fares (where the x axis is the fare and the y axis is the count of people who paid that fare)" or "Is there a relationship between advertising spend and sales?" or "How do height and weight correlate in the dataset? Do not use it for questions that do not have a continuous x axis."
                - Pie Charts: Ideal for showing proportions or percentages within a whole. Use for questions like "What is the market share distribution among different companies?" or "What percentage of the total revenue comes from each product?"
                - Line Graphs: Best for showing trends and distributionsover time. Best used when both x axis and y axis are continuous. Used for questions like "How have website visits changed over the year?" or "What is the trend in temperature over the past decade?". Do not use it for questions that do not have a continuous x axis or a time based x axis.

                Consider these types of questions when recommending a visualization:
                1. Aggregations and Summarizations (e.g., "What is the average revenue by month?" - Line Graph)
                2. Comparisons (e.g., "Compare the sales figures of Product A and Product B over the last year." - Line or Column Graph)
                3. Plotting Distributions (e.g., "Plot a distribution of the age of users" - Scatter Plot)
                4. Trends Over Time (e.g., "What is the trend in the number of active users over the past year?" - Line Graph)
                5. Proportions (e.g., "What is the market share of the products?" - Pie Chart)
                6. Correlations (e.g., "Is there a correlation between marketing spend and revenue?" - Scatter Plot)

                Provide your response in the following format:
                Recommended Visualization: [Chart type or "None"]. ONLY USE THE FOLLOWING CHART TYPES: bar, horizontal_bar, line, pie, scatter, none
                Reason: [Brief explanation for your recommendation]
             
                Do not include any kind of descriptive noun such as plot or graph.
             
                If the recommended visualization is anything other than bar, horizontal_bar, line, pie, scatter, or none, reevaluate your answer and choose one of them.
                '''),
                            ("human", '''
                User question: {question}
                SQL query: {sql_query}
                Query results: {results}
                Recommend a visualization:'''
            ),
        ])

        vis_prompt = prompt.invoke({
            "question": question,
            "sql_query": query,
            "results": results
        })

        response = self.llm.invoke(vis_prompt)

        lines = response.content.split('\n')
        visualization = lines[0].split(': ')[1]
        reason = lines[1].split(': ')[1]

        return visualization, reason
    
    def format_data(self, visualization, results, sql_query, question):
        if visualization == "none":
            return {"data_for_visualization": None}
        
        if visualization == "scatter":
            try:
                return self.format_scatter_data(results, question, sql_query, visualization)
            except Exception as e:
                return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)
        
        if visualization == "bar" or visualization == "horizontal_bar":
            try:
                return self.format_bar_data(results, question)
            except Exception as e:
                return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)
        
        if visualization == "line":
            try:
                return self.format_line_data(results, question, sql_query)
            except Exception as e:
                return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)
            
        return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)

    def format_line_data(self, results, question, query):
        formatted_data = {}

        if len(results[0]) == 2:
            x_values = [str(row[0]) for row in results]
            y_values = [float(row[1]) for row in results]

        # Use LLM to get a relevant label
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, some data, and the SQL Query, provide a concise and relevant label for the data series."),
                ("human", "Question: {question}\n Data (first few rows): {data}\n SQL Query {query}\n\nProvide a concise label for this y axis. For example, if the data is the GDP figures over time, the label could be 'GDP'. If the data is the non paying loans growth, the label could be 'NPL'. If the data is the community bank leverage ratio trend, the label could be 'Community bank leverage ratio'."),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": str(results[:2]),
                "query": query
            })
            label = self.llm.invoke(prompt).content

            formatted_data= {
                "xValues": x_values,
                "yValues": [
                    {
                        "data": y_values,
                        "label": label.strip()
                    }
                ]
            }
            
        elif len(results[0]) == 3:
            x_values = []
            y_values = [{"data": [],
                 "label": ""},
                 {"data": [],
                 "label": ""}
                ]

            for row in results:
                # if type(row[0]) != float and ("-" in row[0] or "/" in row[0]):
                if type(row[0]) != float:
                    x1, y1, y2 = row[0], row[1], row[2]
                else:
                    y1, x1, y2 = row[0], row[1], row[2]

                # if x not in x_values:
                x_values.append(x1)
                y_values[0].get("data").append(y1)
                y_values[1].get("data").append(y2)

            formatted_data = {
                "xValues": x_values,
                "y_values": y_values,
                "yAxisLabel": ""
            }

            # get a relevant lable for the y-axis 
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the numeric values and a label for the y axis."),
                ("human", "Question: {question}\n Data (first few rows): {data}\n SQL Query {query}\n\nProvide a concise label for the second and third values as well as the y axis. For example, if the data is the NPL and GDP figures over time, the first label could be 'NPL', second label 'GDP' and third label 'NPL and GDP over time'. Return the data in the form [value1 label, value2 label, y axis label]"),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": str(results[:2]),
                "query": query
            })
            label = self.llm.invoke(prompt)
            try:
                label = ast.literal_eval(label.content)
                y_values[0]["label"] = label[0]
                y_values[1]["label"] = label[1]
                formatted_data["yAxisLabel"] = label[2]
                print("label", formatted_data)
            except:
                print(label)
                print("error with vis")
       
        return {"formatted_data": formatted_data}

    def format_scatter_data(self, results, question, sql_query, visualization):
        if isinstance(results, str):
            results = eval(results)

        formatted_data = {"series": [], "xLabel": "", "yLabel": ""}
        

        if len(results[0]) == 2:
            formatted_data["series"].append({
                "data": [
                    {"x": float(x), "y": float(y), "id": i+1}
                    for i, (x, y) in enumerate(results)
                ]
            })
            
            # get a relevant lable for the y-axis 
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, a query, some data, and the visualization type, provide a concise and relevant label for the x axis and the y axis."),
                ("human", "Question: {question}\n Query:{query} \n Data (first few rows): {data}\n Visualization type:{visualization}\n\nProvide a concise label for the x axis and the y axis. For example, if the data is the NPL and GDP, the x axis label could be 'NPL' and the y axis label could be 'GDP'. Return the data in the form '[xLabel, yLabel]'")
            ])
            prompt_val = prompt.invoke({
                "question": question,
                "data": results,
                "query": sql_query,
                "visualization": visualization
            })
            label = self.llm.invoke(prompt_val).content
            label = ast.literal_eval(label)

            formatted_data["xLabel"] = label[0]
            formatted_data["yLabel"] = label[1]


        elif len(results[0]) == 3:
            print("here")
            entities = {}
            for row in results:
                # Determine which item is the label (string not convertible to float and not containing "/")
                if isinstance(row[0], str) and type(row[0]) != float and "-" not in row[0]:
                    label, x, y = row[0], row[1], row[2]
                else:
                    x, label, y = row[0], row[1], row[2]

                if label not in entities:
                    entities[label] = []
                entities[label].append({"x": x, "y": y, "id": len(entities[label])+1})
            
            for label, data in entities.items():
                formatted_data["series"].append({
                    "data": data,
                    "label": label
                })

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, a query, some data, and the visualization type, provide a concise and relevant label for the x axis and the y axis."),
                ("human", "Question: {question}\n Query:{query} \n Data (first few rows): {data}\n Visualization type:{visualization}\n\nProvide a concise label for the x axis and the y axis. For example, if the data is the NPL and GDP, the x axis label could be 'NPL' and the y axis label could be 'GDP'. Return the data in the form '[xLabel, yLabel]'"),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": results,
                "query": sql_query,
                "visualization": visualization
            })

            label = self.llm.invoke(prompt).content
            label = ast.literal_eval(label)
            formatted_data["xLabel"] = label[0]
            formatted_data["yLabel"] = label[1]

        else:
            raise ValueError("Unexpected data format in results")                

        return {"formatted_data": formatted_data}

    def format_bar_data(self, results, question):
        # results = data
        if isinstance(results, str):
            results = eval(results)

        if len(results[0]) == 2:
            # Simple bar chart with one series
            xValues = [str(row[0]) for row in results]
            data = [float(row[1]) for row in results]
            
            # Use LLM to get a relevant label
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the data series."),
                ("human", "Question: {question}\nData (first few rows): {data}\n\nProvide a concise label for this y axis. For example, if the data is the sales figures for products, the label could be 'Sales'. If the data is the population of cities, the label could be 'Population'. If the data is the revenue by region, the label could be 'Revenue'."),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": data
            })
            yLabel = self.llm.invoke(prompt).content
            
            yValues = [{"data": data, "label": yLabel}]

        elif len(results[0]) == 3:
            # Grouped bar chart with multiple series
            categories = set(row[1] for row in results) #eg. date
            xValues = list(categories) #eg. date
            entities = set(row[0] for row in results) #eg. gdp, npl
            yValues = [] #eg. more numerical like um the amount of gdp
            # for each entity get all the data for that entity
            for entity in entities:
                entity_data = [float(row[2]) for row in results if row[0] == entity]
                yValues.append({"data": entity_data, "label": str(entity)})
        else:
            raise ValueError("Unexpected data format in results")

        formatted_data = {
            "xValues": xValues,
            "yValues": yValues
        }

        return {"formatted_data": formatted_data}

    def format_other_visualizations(self, visualization, question, results, sql_query):
        instructions = graph_instructions[visualization]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Data expert who formats data according to the required needs. You are given the question asked by the user, it's sql query, the result of the query and the format you need to format it in."),
            ("human", 'For the given question: {question}\n\nSQL query: {sql_query}\n\Result: {results}\n\nUse the following example to structure the data: {instructions}. Just give the json string. Do not format it')
        ])
        response = self.llm.invoke(prompt, question=question, sql_query=sql_query, results=results, instructions=instructions)
            
        try:
            formatted_data_for_visualization = json.loads(response)
            return {"formatted_data_for_visualization": formatted_data_for_visualization}
        except json.JSONDecodeError:
            return {"error": "Failed to format data for visualization", "raw_response": response}


@tool
# def visualization_tool():
def visualization_tool(state: Annotated[State, InjectedState],
     tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:

    """
    Chooses the type of visualization that the question requires and formats data for visualization. 
    This tool should be called ONLY after the sql_tool.

    Args: 
        results: state["results"] as stored in the state WITH NO MUTATIONS REORDERING FOR THE INTENDED VISUALIZATION.  
        sql_query: state["sql_query"] as stored in the state with no mutations 
        question: the user's question 
    """

    question=state.get("user_question", None),
    sql_query = state.get("sql_query", None),
    sql_results=state.get("sql_results", None)

    if not sql_results:
        print("NO SQL RES")
        return Command(update={
            "messages": [ToolMessage("No data available for visualization", tool_call_id=tool_call_id)],
            "visualization": "none",
            "visualization_reason": "none",
            "formatted_data_for_visualization": {}
        })

    vis = VisualizationTool("gpt-3.5-turbo", 0)
    visualization, reason = vis.visualization_type(
        question=question,
        query = sql_query,
        results=sql_results
    )
    visualization = visualization.strip().lower()

    visualization = "line"

    formatted_data = vis.format_data(visualization=visualization,
                                    results=sql_results,sql_query=sql_query,
                                    question=question)
    
    return Command(update={
        "messages": [ToolMessage("Success", tool_call_id=tool_call_id)],
        "visualization": visualization,
        "visualization_reason": reason,
        "formatted_data_for_visualization": formatted_data
    })

# visualization_tool()