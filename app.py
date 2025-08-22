from src.chatbot import ChatBot

c = ChatBot()
user_id = "u_42"
session_id = "s_2025_08_21_06"
thread_id = f"{user_id}:{session_id}"

res = c.respond("compare the GDP and exchange rate throughout 2024", user_id=user_id, session_id=session_id)
print(res)

