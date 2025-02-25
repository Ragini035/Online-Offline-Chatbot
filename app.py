import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Memory for Context Retention
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Langsmith Tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Ollama"

# Set Streamlit Page Config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Chatbot - Ask Anything!")

# Sidebar Settings
st.sidebar.title("⚙️ Settings")
engine = st.sidebar.selectbox("🧠 Select Model", ["gemma2:2b", "llama3.2", "mistral"])
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("📏 Max Tokens", 50, 300, 150)

# Prompt Template (Now uses chat history for context)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Maintain conversation history for context."),
    ("user", "Previous conversation: {chat_history}\n\nUser: {question}")
])

# Create LLMChain with memory
llm = Ollama(model=engine)
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory  # Attach memory to store chat history
)

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    st.session_state.memory.clear()  # Reset memory for new conversation
    st.rerun()
