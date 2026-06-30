import os
import sys

# Add the parent directory to sys.path so we can import from chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.agent import get_agent
from langchain_core.messages import HumanMessage

def run_chat():
    print("Initializing Custom ACS880 Drive Failure Chatbot...")
    try:
        app = get_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return
        
    print("\nChatbot is ready! Type 'exit' or 'quit' to stop.")
    
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue
            
        # Append user message
        messages.append(HumanMessage(content=user_input))
        
        state = {
            "messages": messages,
            "query": user_input
        }
        
        try:
            # Invoke the custom graph
            final_state = app.invoke(state)
            
            # The agent returns the full updated messages list.
            messages = final_state["messages"]
            
            # Extract the last message (which is from the AI)
            ai_message = messages[-1]
            print(f"\nBot: {ai_message.content}")
            
        except Exception as e:
            print(f"\nAn error occurred during chat: {e}")

if __name__ == "__main__":
    run_chat()
