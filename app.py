import os
import streamlit as st
import uuid
import tempfile
import shutil
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

from graph import graph, process_file

# --- Configuration ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("LangGraph RAG Chatbot")

# --- Sidebar ---
st.sidebar.header("Configuration")

# Thread ID Management
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

thread_id_input = st.sidebar.text_input("Thread ID", value=st.session_state.thread_id)
if thread_id_input != st.session_state.thread_id:
    st.session_state.thread_id = thread_id_input
    # Clear chat history visual on thread switch (optional, but good for clarity)
    st.session_state.messages = [] 
    st.toast(f"Switched to thread: {thread_id_input}")

st.sidebar.divider()

# File Uploader
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx", "doc"])

if uploaded_file is not None:
    # Check if this file is new
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner("Processing document..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process the file (clears vector store and re-indexes)
                process_file(tmp_path)
                st.session_state.last_uploaded_file = uploaded_file.name
                st.sidebar.success(f"Loaded {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
else:
    # If file removed using the 'x', we might want to optionally clear the vector store, 
    # but the requirement says "clears ... when a new file is added". We can leave it for now.
    pass

st.sidebar.divider()

# Feedback Placeholder (LangSmith)
st.sidebar.header("Feedback")
if st.sidebar.button("Log Positive Feedback (+1)"):
    st.sidebar.success("Feedback logged to LangSmith! (Placeholder)")
    # Code to log feedback to LangSmith would go here:
    # client.create_feedback(run_id=..., key="user_score", score=1.0)

if st.sidebar.button("Log Negative Feedback (-1)"):
    st.sidebar.warning("Feedback logged to LangSmith! (Placeholder)")

# --- Chat Interface ---

# Initialize chat history in session state for display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to display history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        # Prepare configuration
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # We need to stream the output to see the steps
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream events from the graph
        # Using .stream() with the inputs
        inputs = {"messages": [HumanMessage(content=prompt)]}
        
        try:
            # We use the graph.stream to get updates. 
            # Note: LangGraph stream yields dictionary updates of the state.
            for event in graph.stream(input=inputs, config=config, stream_mode="values"):  # type: ignore
                # "values" mode yields the full state at each step suitable for finding the final message
                # Or we can use "updates" to see node outputs.
                
                # We want the final generation from the "generate" node.
                # But we also want to see intermediate steps (handled by callback? LangGraph with callbacks is supported).
                pass
            
            # After stream completes, get final state
            snapshot = graph.get_state(config=config)  # type: ignore  # Get the final state snapshot after execution
            if snapshot.values and "messages" in snapshot.values:
                last_msg = snapshot.values["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    full_response = last_msg.content
                    response_placeholder.markdown(full_response)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error during execution: {e}")
