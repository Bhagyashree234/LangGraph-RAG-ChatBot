# LangGraph RAG Chatbot 

A production-ready RAG (Retrieval-Augmented Generation) Chatbot built using **LangChain**, **LangGraph**, and **Streamlit**. This project leverages **Groq** for high-speed LLM inference and **LangSmith** for deep tracing and monitoring.

##  Project Goal (Overview)
Build a state-aware chatbot capable of persistent conversations over uploaded documents.
- **LLM**: Groq (`llama-3.3-70b-versatile`)
- **Embeddings**: Local HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (Local/Ephemeral updates on new file upload)
- **Orchestration**: LangGraph `StateGraph`
- **Memory**: Persistent thread-based memory using `MemorySaver`
- **Tracing**: LangSmith traces every step of the LangGraph execution and stores conversation threads using unique thread IDs, enabling debugging, performance monitoring, and evaluation of LLM applications.

## Prerequisites
- Python 3.11+:
- Groq API Key: 
- LangChain API Key:

## Setup Instructions

1. **Install Dependencies**
   Ensure you are in the project root directory.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Create a `.env` file (or rename `.env.example`). Add your API keys:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=langgraph-rag-chatbot
   LANGCHAIN_API_KEY=your_langchain_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Run the Application**
   Start the Streamlit interface:
   ```bash
   streamlit run app.py
   ```
![image](https://github.ibm.com/user-attachments/assets/7065c70b-f362-4e27-9d09-d1a594aa6703)


##  Workflow & Architecture

### 1. User Interface (`app.py`)
- **Sidebar**: Manage "Thread ID" (sessions) and upload documents (PDF/TXT/DOCX).
- **Session State**: Handles chat history display and file changes.

### 2. The Logic Graph (`graph.py`)
The application uses a `StateGraph` with the following nodes:
- **`retrieve` Node**:
  - Uses `FAISS` vector store to find relevant chunks from the uploaded document.
  - Embeddings are generated locally using `HuggingFaceEmbeddings`.
- **`generate` Node**:
  - Takes the retrieved context and chat history.
  - Calls `ChatGroq` (`llama-3.3-70b-versatile`) to generate a response.
- **Persistence (`checkpointer`)**:
  - Uses `MemorySaver` to save the state of the graph.
  - Allows resuming conversations by switching the **Thread ID**.

### 3. Usage Flow
1. **Enter Thread ID**: Use a unique ID to start a fresh session or an existing ID to resume.
2. **Upload Document**: Upload a PDF or TXT file. The system indexes it immediately.
   - *Note*: Uploading a new file clears the previous vector store index.
3. **Ask Questions**: Type your query. The bot retrieves context and answers.
4. **Monitor**: Check [LangSmith](https://smith.langchain.com/) to see the full trace of your request (Retrieval -> Generation).

## Project Structure
- `app.py`: Streamlit frontend handling UI and user inputs.
- `graph.py`: Defines the LangGraph State, Nodes, and Vector Store logic.
- `requirements.txt`: Python package dependencies.
- `.env`: Environment variables (API Keys).
- `README.md`: Project documentation.

## Output:
Outputs:
- Context-aware AI-generated responses
- Chat history per thread
- Error messages if no document is uploaded

![image](https://github.ibm.com/user-attachments/assets/187fb784-daa8-4a00-a9b9-aad11bd062e7)
<img width="975" height="468" alt="image" src="https://github.com/user-attachments/assets/f4122d5e-c998-4290-937b-c204c0e10608" />


