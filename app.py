import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Presented By Hussain! 313")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Keep only the last 5 messages for context (to prevent long memory issues)
    result = model.invoke(st.session_state.chat_history[-5:])
    st.session_state.chat_history.append(AIMessage(content=result.content))

    # Use chat bubbles
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(result.content)

# Display Chat History
st.subheader("Chat History")
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)
