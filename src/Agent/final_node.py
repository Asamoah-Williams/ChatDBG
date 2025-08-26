from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from src.utils.load_tools_config import LoadToolsConfig
from langchain_openai import ChatOpenAI
from src.Agent.State import State
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

TOOLS_CFG = LoadToolsConfig()



def finalizer(state: State):
    """Creates a concise final answer based on the graph state and messages"""
    llm = ChatOpenAI(model=TOOLS_CFG.primary_agent_llm, 
                    temperature=TOOLS_CFG.primary_agent_llm_temperature, streaming=True)
    
    question = state.get("user_question")
    visualization = state.get("visualization", None)
    visualization_reason = state.get("visualization_reason", None)
    messages = state.get("messages", None)
    response = messages[-1].content

    trimmed_messages = trim_messages(
        messages,
        max_tokens=500,
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            Your name is CHAT DBG, a friendly finance expert finalizing the answer for the user. Write a concise, grounded response considering ONLY the user question {question} and the formulated response {response}.
            IF THE VISUALIZATION VALUE: {visualization} IS NOT NONE OR ANY OTHER VALUE INDICATING NO VISUALIZATION, include a 50 word concise and accurate summary of the visualization using the visualization name {visualization}. If the visualization value {visualization} is empty do not include any summary or mention the lack of a visualization. The summary of the visualization should always come after the answer to the user's prompt. If something in the user prompt is unclear, state it briefly. If the answer is not sufficient for the user's question state it briefly. Do not include any links to images.
        """),
        ("human",
            """
        Messages: {messages} \n
        Visualization: {visualization} \n
        Visualization Reason: {visualization_reason} \n
        Question: {question} \n
        Response: {response}
        Provide a concise, grounded answer to my question.
        """ )
    ])

    prompt = prompt.format_messages(
        messages = trimmed_messages,
        visualization = visualization,
        visualization_reason = visualization_reason,
        question = question,
        response = response
    )

    final_answer = llm.invoke(prompt)

    return Command(update={
        "answer": final_answer.content,
        "answer_done": True
    })
