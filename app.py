from src.chatbot import ChatBot
from datetime import datetime
import ast
import json

# try all    except error create a new session

c = ChatBot()
user_id = "u_89"
session_id = "s_2025_08_25_13"
thread_id = f"{user_id}:{session_id}"

res = c.respond("i want a line graph showing npl and gdp from 2018 to 2022", user_id=user_id, session_id=session_id)
print(res)