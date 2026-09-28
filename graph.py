import os
from typing import Annotated, List, TypedDict, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global State for Vector Store (In-Memory for this session)
# In production, use a persistent vector store (e.g., Chroma, Pinecone, or FAISS saved to disk)
VECTOR_STORE = None
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_file(file_path: str):
    """
    Ingests a file, splits it, and updates the global VECTOR_STORE.
    This function should be called when a new file is uploaded.
    """
    global VECTOR_STORE
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select loader
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx") or file_path.lower().endswith(".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        # Default to text loader for other formats or txt
        loader = TextLoader(file_path)
    
    docs = loader.load()
    if not docs:
        print("No documents loaded.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create or Overwrite Vector Store (Per user requirement: "clears the vector store only when a new file is added")
    # This effectively resets it.
    VECTOR_STORE = FAISS.from_documents(documents=splits, embedding=EMBEDDINGS)
    print(f"Vector store updated with {len(splits)} chunks from {file_path}")

# --- Graph Definition ---

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # We can add a 'context' key if we want specifically retrieved docs, 
    # but for simplicity we'll inject context into the prompt in the generate node.
    context: Optional[str]

def retrieve(state: State):
    """
    Retrieve documents based on the last user message.
    """
    global VECTOR_STORE
    
    # If no vector store is loaded, we can't retrieve.
    if VECTOR_STORE is None:
        return {"context": "No document loaded. Please upload a document to ask questions."}
    
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    
    # Retrieve top k documents
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    # Format context
    context = "\n\n".join([d.page_content for d in docs])
    return {"context": context}

def generate(state: State):
    """
    Generate answer using Groq LLM.
    """
    context = state.get("context", "")
    messages = state["messages"]
    
    # System prompt to enforce RAG
    if context and context != "No document loaded. Please upload a document to ask questions.":
        system_msg_content = (
            "You are a helpful assistant. Use the following pieces of context to answer the user's question.\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
            f"Context:\n{context}"
        )
    else:
        system_msg_content = "You are a helpful assistant. No context is currently available."
        if context: # Propagate the missing doc message if it's the specific specific "No document loaded" string
             system_msg_content += f" {context}"

    # We filter out the SystemMessages from history to avoid stacking them, or just prepend the new one.
    # A simple way is to invoke the LLM with the system message + history.
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # Construct the full prompt messages
    prompt_messages = [SystemMessage(content=system_msg_content)] + messages
    
    response = llm.invoke(prompt_messages)
    
    return {"messages": [response]}

def init_graph():
    """
    Initializes and compiles the StateGraph.
    """
    workflow = StateGraph(State)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# Expose the graph
graph = init_graph()
