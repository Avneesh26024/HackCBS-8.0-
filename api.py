# api.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import the necessary components and global objects from your existing files
from query_tool import DataEngine
from embedding_manager import EmbeddingManager
from langchain_core.messages import HumanMessage, AIMessage

# Import the pre-compiled graph 'app' and the global state objects from main.py
# This is the key to linking the API to your agent
from main import (
    app,
    data_engine as global_data_engine,
    embedding_manager as global_embedding_manager
)

# --- 1. Initialize FastAPI App & CORS ---
app_api = FastAPI(
    title="AI-Driven DB RAG & Analytics API",
    description="API for chatting with your database."
)

# Enable CORS for all domains (for web frontends)
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- 2. Define API Request/Response Models ---
class UploadRequest(BaseModel):
    source: str


class UploadResponse(BaseModel):
    status: str
    message: str
    db_structure: dict


class Message(BaseModel):
    type: str  # "human" or "ai"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    messages: List[Message]


class ModelGeneration3DRequest(BaseModel):
    source: str


# --- 3. Define API Endpoints ---

@app_api.post("/upload_db", response_model=UploadResponse)
def upload_db(request: UploadRequest):
    """
    Loads a new data source (SQL DB, CSV, Excel) and runs the
    embedding process. This prepares the agent for chatting.
    """
    global global_data_engine, global_embedding_manager

    # If the user wants the test DB, create it if it doesn't exist
    if request.source == "sqlite:///elaborate_test.db" and not os.path.exists("elaborate_test.db"):
        print("--- 'elaborate_test.db' not found. Creating it... ---")
        try:
            from embedding_manager import main as create_dummy_db
            create_dummy_db()
            print("--- Dummy 'elaborate_test.db' created. ---")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create dummy db: {e}")

    print(f"--- Loading data source: {request.source} ---")

    # We must re-assign the *global* variable that the agent graph uses
    global_data_engine = DataEngine()

    if not global_data_engine.load(request.source):
        raise HTTPException(status_code=400, detail="Failed to load data source.")

    print("--- Building vector stores ---")
    global_embedding_manager = EmbeddingManager(global_data_engine)
    global_embedding_manager.build_vector_stores()

    print("--- RAG system is online ---")

    # Get the structure to send back to the client
    structure = global_data_engine.get_database_structure_dict()

    return {
        "status": "success",
        "message": "Data source loaded and vector stores built.",
        "db_structure": structure
    }


@app_api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Handles a single turn of the chat conversation.
    Receives the full message history and returns the new history.
    """
    global global_embedding_manager, app

    if not global_embedding_manager:
        raise HTTPException(status_code=400, detail="Database not uploaded. Please call /upload_db first.")

    # Convert Pydantic models to LangChain messages
    lc_messages = []
    for msg in request.messages:
        if msg.type == "human":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            lc_messages.append(AIMessage(content=msg.content))

    # We must reset the computed fields on each new query
    # This prevents state from "leaking" between runs
    current_state = {
        "messages": lc_messages
    }

    # Invoke the graph
    print(f"--- Invoking graph for query: {lc_messages[-1].content} ---")
    final_state = app.invoke(current_state)
    print("--- Graph invocation complete ---")

    # Convert final LangChain messages back to Pydantic models
    response_messages = []
    for msg in final_state['messages']:
        response_messages.append(Message(type=msg.type, content=msg.content))

    return {"messages": response_messages}


@app_api.post("/3d_generate")
def generate_3d_model(request: ModelGeneration3DRequest):
    """
    Generates a structured database schema JSON from a data source connection string.
    This is used for 3D visualization on the frontend.
    """
    print(f"--- Generating 3D schema for source: {request.source} ---")
    try:
        engine = DataEngine()
        if not engine.load(request.source):
            raise HTTPException(status_code=400, detail="Failed to load data source.")

        manager = EmbeddingManager(engine)
        schema_json = manager.generate_structured_schema_json()

        if not schema_json:
            raise HTTPException(status_code=500, detail="Failed to generate structured schema from the data source.")

        return schema_json
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error in /3d_generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("--- Starting FastAPI server ---")
    print("Visit http://127.0.0.1:8000/docs for API documentation")
    uvicorn.run("api:app_api", host="127.0.0.1", port=8000, reload=True)