from src.chatbot import ChatBot

c = ChatBot()
user_id = "u_44"
session_id = "s_2025_08_24_04"
thread_id = f"{user_id}:{session_id}"

res = c.respond("i want a line graph comparing exchange rate and npl between 2014 and 2016"
, user_id=user_id, session_id=session_id)
print(res)

