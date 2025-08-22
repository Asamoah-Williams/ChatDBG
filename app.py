from src.chatbot import ChatBot

c = ChatBot()
user_id = "u_42"
session_id = "s_2025_08_22_18"
thread_id = f"{user_id}:{session_id}"

res = c.respond("what is happening in finance today", user_id=user_id, session_id=session_id)
print(res)

