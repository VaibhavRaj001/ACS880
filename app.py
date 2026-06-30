import streamlit as st
import os
import sys
import time
from langchain_core.messages import HumanMessage, AIMessage

# Add the current directory to sys.path so we can import from chatbot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.agent import get_agent

# Configure Streamlit page
st.set_page_config(
    page_title="ACS880 Diagnostics", 
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .reportview-container {
        background: #f4f6f9;
    }
    .sidebar .sidebar-content {
        background: #2b3035;
        color: white;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .title-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #003b5c;
        font-weight: 600;
        margin-bottom: 0rem;
    }
    .subtitle-text {
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Global Chat Layout */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }

    /* Human Message (Right) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        background-color: rgba(0, 132, 255, 0.15); /* Adaptive Blue Tint */
        margin-left: auto;
        margin-right: 0;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        text-align: left; /* Keep text left aligned inside bubble */
    }

    /* AI Message (Left) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        flex-direction: row;
        background-color: rgba(128, 128, 128, 0.1); /* Adaptive Grey Tint */
        margin-right: auto;
        margin-left: 0;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        text-align: left; 
    }

    /* Content styling */
    [data-testid="stChatMessageContent"] {
        flex-grow: 1;
        padding: 0 1rem;
    }
""", unsafe_allow_html=True)

# Initialize agent
@st.cache_resource
def load_agent():
    return get_agent()

try:
    app = load_agent()
except Exception as e:
    st.error(f"Failed to load the diagnostic agent. Please ensure the vector database is initialized. Error: {e}")
    st.stop()

# Sidebar UI
with st.sidebar:
    
    st.markdown("### Support Diagnostics")
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Enter the specific fault code (e.g., `F5001`).
    2. Or describe the symptom (e.g., `Motor stalling during acceleration`).
    
    The assistant will retrieve the relevant technical documentation and provide a structured troubleshooting guide.
    """)
    st.markdown("---")
    st.caption("Powered by LangGraph & Groq")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your ACS880 technical support assistant. Please provide a fault code or describe the issue you are experiencing with the drive."}
        ]
        st.rerun()

# Main UI
st.markdown('<h1 class="title-text">⚙️ ACS880 Drive Diagnostics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Your AI-powered technical support engineer.</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your ACS880 technical support assistant. Please provide a fault code or describe the issue you are experiencing with the drive."}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What causes an F5001 fault?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg["content"] != st.session_state.messages[0]["content"]:
            langchain_messages.append(AIMessage(content=msg["content"]))
            
    state = {
        "messages": langchain_messages,
        "query": prompt
    }
    
    with st.chat_message("assistant"):
        with st.spinner("Diagnosing issue..."):
            try:
                final_state = app.invoke(state)
                ai_message = final_state["messages"][-1].content
                
                def stream_response():
                    for word in ai_message.split(" "):
                        yield word + " "
                        time.sleep(0.04)
                        
                st.write_stream(stream_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_message})
            except Exception as e:
                st.error(f"An error occurred during diagnosis: {e}")
