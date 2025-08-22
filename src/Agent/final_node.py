from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from src.utils.load_tools_config import LoadToolsConfig
from langchain_openai import ChatOpenAI
from src.Agent.State import State

TOOLS_CFG = LoadToolsConfig()



def finalizer(state: State):
    """Creates a concise final answer based on the graph state and messages"""
    llm = ChatOpenAI(model=TOOLS_CFG.primary_agent_llm, 
                    temperature=TOOLS_CFG.primary_agent_llm_temperature, streaming=True)

    visualization = state.get("visualization", None)
    visualization_reason = state.get("visualization_reason", None)
    messages = state.get("messages", None)
    data = state.get("formatted_data_for_visualization", None)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            You are a friendly finance expert finalizing the answer for the user. Write a concise, grounded response using ONLY the state messages {messages}. IF THE VISUALIZATION {visualization} VALUE IS NOT NONE OR ANY OTHER VALUE INDICATING NO VISUALIZATION, include a 150 word concise and accurate summary of the visualization using the visualization name {visualization}, the visualization reason {visualization_reason}, and the data to be visualized {data}. The summary of the visualization should always come after the answer to the user's prompt. If something in the user prompt is unclear, state it briefly. If the answer is not sufficient for the user's question state it briefly. Do not include any links to images.
        """),
        ("human",
            """
        Messages: {messages} \n
        Visualization: {visualization} \n
        Visualization Reason: {visualization_reason} \n
        Data: {data} \n
        Provide a concise, grounded answer to my question.
        """ )
    ])

    prompt = prompt.format_messages(
        messages = messages,
        visualization = visualization,
        visualization_reason = visualization_reason,
        data = data
    )

    final_answer = llm.invoke(prompt)
    # print("final answer type" ,type(final_answer.content))


    return Command(update={
        "answer": final_answer.content,
        "answer_done": True
    })
