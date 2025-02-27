import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API Key for Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize Memory for Context Retention
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set Streamlit Page Config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Chatbot - Ask Anything!")

# Sidebar Settings
st.sidebar.title("⚙️ Settings")
model_type = st.sidebar.radio("Select Mode", ["Offline (Ollama)", "Online (Groq)"])

# Set LLM Model
if model_type == "Online (Groq)":
    engine = st.sidebar.selectbox("🧠 Select Groq Model", ["mixtral-8x7b-32768", "llama3-8b", "gemma-7b"])
    llm = ChatGroq(model_name=engine)
else:
    engine = st.sidebar.selectbox("🧠 Select Ollama Model", ["gemma2:2b", "llama3.2", "mistral"])
    llm = Ollama(model=engine)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Maintain conversation history for context."),
    ("user", "Previous conversation: {chat_history}\n\nUser: {question}")
])

# Create LLMChain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory
)

# User Input Box at the Bottom
user_input = st.chat_input("Type your question here...")

# Process User Input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = chain.run({"question": user_input, "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]})

    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display Chat History
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.rerun()

