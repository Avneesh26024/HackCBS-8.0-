# main.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import re
from typing import List, TypedDict, Dict, Any, Annotated
import operator
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from query_tool import DataEngine
from embedding_manager import EmbeddingManager
from image_result import save_plot

# Load environment variables (for Google API key)
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY not found. Please set it in .env file.")
    exit()

# --- 1. Setup Models, DataEngine, and EmbeddingManager ---

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
vision_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

data_engine = DataEngine()
embedding_manager = None  # Will be initialized after data is loaded


# --- 2. Define Agent State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    rag_context: str
    sql_query: str
    sql_result_str: str
    sql_result_df: Any  # The raw DataFrame for analysis
    current_query: str

    # NEW Flexible Analysis Fields
    analysis_code: str  # Code for plotting OR stats
    analysis_result: str  # Result from a statistical calculation
    image_path: str  # Path to a saved plot
    image_analysis: str  # Verdict from analyzing a plot


# --- 3. Define Agent Nodes ---

def classify_intent_node(state: AgentState):
    """
    Node 1 (NEW ENTRY POINT): Classifies the user's intent.
    Routes to RAG/SQL flow or answers directly from chat history.
    """
    print("Node: classify_intent_node")
    messages = state['messages']

    # If this is the very first message, it must be a new query.
    if len(messages) <= 1:
        print("Routing: new_data_query (first message)")
        return "new_data_query"

    current_query = messages[-1].content

    # Format history for the LLM
    history_str = "\n".join(
        [f"{m.type}: {m.content}" for m in messages[:-1]]  # All messages *except* the last one
    )

    prompt = f"""You are an intent classification agent.
Your job is to decide if the "User's Latest Query" can be answered *only* from the "Chat History" or if it's a new query that requires fetching data from a database.

Chat History:
{history_str}

User's Latest Query: "{current_query}"

-   If the query is a simple follow-up, a greeting, or asking about the *previous* response (e.g., "thanks", "what was that number?", "can you explain that plot?"), respond with: **answer_from_history**
-   If the query is a new question asking for data, analysis, or a plot (e.g., "how many users?", "plot sales vs time", "what's the correlation?"), respond with: **new_data_query**

Respond with only one of the two options.
"""

    response = model.invoke(prompt)
    route = response.content.strip().lower()

    if "answer_from_history" in route:
        print("Routing: answer_from_history")
        return "answer_from_history"
    else:
        print("Routing: new_data_query")
        return "new_data_query"


def set_current_query_node(state: AgentState) -> dict:
    """
    Node 1.B: Sets the current_query in the state.
    This is the simple action the old 'intent_node' used to do.
    """
    print("Node: set_current_query_node")
    last_message = state['messages'][-1]
    return {"current_query": last_message.content}


def rag_node(state: AgentState) -> dict:
    """ Node 2: Retrieves RAG context. """
    print("Node: rag_node")
    query = state['current_query']
    if embedding_manager:
        context = embedding_manager.query_rag_context(query)
        return {"rag_context": context}
    else:
        print("Error: EmbeddingManager not initialized.")
        return {"rag_context": "Error: EmbeddingManager not initialized."}


def sql_generation_node(state: AgentState) -> dict:
    """ Node 3: Generates the SQL query. """
    print("Node: sql_generation_node")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL database assistant.
Your task is to generate a single, valid SQL query to answer the user's question, based on the provided database context.
Only output the SQL query. Do not include any other text, explanations, or markdown.

Database Context:
{context}"""),
        ("human", "User's Question: {query}")
    ]).format(context=state['rag_context'], query=state['current_query'])

    response = model.invoke(prompt)
    sql_query = response.content.strip().replace("```sql", "").replace("```", "")
    print(f"Generated SQL: {sql_query}")
    return {"sql_query": sql_query}


def sql_execution_node(state: AgentState) -> dict:
    """ Node 4: Executes the SQL query using DataEngine. """
    print("Node: sql_execution_node")
    query = state['sql_query']
    result_df = data_engine.query(query)

    if result_df is not None and not result_df.empty:
        result_str = result_df.to_string()
    elif result_df is not None:
        result_str = "Query executed successfully (no rows returned)."
    else:
        result_str = "Error: Query failed to execute."

    print(f"SQL Result: {result_str}")
    return {
        "sql_result_str": result_str,
        "sql_result_df": result_df
    }


def router_node(state: AgentState):
    """ Node 5: LLM-based router for complex analysis. """
    print("Node: router_node")
    if state['sql_result_df'] is None or state['sql_result_df'].empty:
        print("Routing: No data from SQL, skipping analysis.")
        return "final_response_branch"
    query = state['current_query']
    data_head = state['sql_result_df'].head().to_string()
    prompt = f"""You are a routing agent. Your job is to decide if a user's query requires complex analysis (like statistics, plotting, anomaly detection) or if the raw data from the SQL query is a sufficient answer.

User Query: "{query}"
Data from SQL:
{data_head}

Queries that need "complex_analysis":
- "plot the..."
- "visualize..."
- "what is the correlation..."
- "find anomalies in..."
- "calculate the skewness of..."
- "compare age and salary"

Queries that are "simple_response":
- "show me all users"
- "how many orders are there"
- "list the products"

Based on the User Query, what is the next step?
Respond with only one word: 'complex_analysis' or 'simple_response'.
"""
    response = model.invoke(prompt)
    route = response.content.strip().lower()
    if "complex_analysis" in route:
        print("Routing: complex_analysis_branch")
        return "complex_analysis_branch"
    else:
        print("Routing: final_response_branch")
        return "final_response_branch"


def analysis_code_generation_node(state: AgentState) -> dict:
    """ Node 6.A: Generates Python code for visualization OR statistical analysis. """
    print("Node: analysis_code_generation_node")
    query = state['current_query']
    df_head = state['sql_result_df'].head().to_string()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert Python data scientist.
You will be given a user's query and the head of a pandas DataFrame (named 'df').
Your task is to write Python code to perform the analysis needed to answer the query.
You must decide whether to:
1.  **Perform a Calculation:** (e.g., correlation, skewness, summary).
2.  **Generate a Plot:** (e.g., histogram, scatter plot, bar chart).

YOUR CODE MUST DEFINE TWO VARIABLES:
1.  `is_plot`: A boolean (True if you are making a plot, False otherwise).
2.  EITHER:
    - `fig`: A `matplotlib.figure.Figure` object (if `is_plot = True`).
    - `analysis_result`: A string or number with the result (if `is_plot = False`).

CRITICAL RULES:
- The DataFrame is already loaded in a variable named `df`.
- If plotting, assign the figure to `fig`. DO NOT call `plt.show()`.
- If calculating, assign the result to `analysis_result`.
- Just output the raw Python code, no explanations or markdown.
"""),
        ("human", f"User Query: {query}\n\nDataFrame Head:\n{df_head}")
    ]).format(query=query, df_head=df_head)
    response = model.invoke(prompt)
    code = response.content.strip().replace("```python", "").replace("```", "")
    print(f"--- Generated Analysis Code ---\n{code}\n-----------------------------")
    return {"analysis_code": code}


def execute_analysis_node(state: AgentState) -> dict:
    """ Node 6.B: Executes the analysis code. """
    print("Node: execute_analysis_node")
    code = state['analysis_code']
    df = state['sql_result_df']
    exec_scope = {"plt": plt, "pd": pd, "df": df, "is_plot": None, "fig": None, "analysis_result": None}
    try:
        exec(code, exec_scope)
        is_plot = exec_scope.get('is_plot')
        if is_plot is True:
            fig = exec_scope.get('fig')
            if fig and isinstance(fig, plt.Figure):
                image_path = save_plot(fig, state['current_query'])
                return {"image_path": image_path, "analysis_result": None}
            else:
                print("❌ Code set is_plot=True but 'fig' was not a valid Figure.")
                return {"analysis_result": "Error: Plot generation failed.", "image_path": None}
        elif is_plot is False:
            result = exec_scope.get('analysis_result')
            print(f"Statistical Result: {result}")
            return {"analysis_result": str(result), "image_path": None}
        else:
            print("❌ Analysis code did not set 'is_plot' boolean variable correctly.")
            return {"analysis_result": "Error: Analysis code was invalid.", "image_path": None}
    except Exception as e:
        print(f"❌ Error executing analysis code: {e}")
        return {"analysis_result": f"Error in analysis code: {e}", "image_path": None}


def analysis_router_node(state: AgentState):
    """ Node 6.C: Checks output of previous step and routes. """
    print("Node: analysis_router_node")
    if state.get("image_path"):
        print("Routing: analyze_image_branch")
        return "analyze_image_branch"
    else:
        print("Routing: final_response_branch")
        return "final_response_branch"


def analyze_image_node(state: AgentState) -> dict:
    """ Node 6.D: Sends the saved image to Gemini for analysis. """
    print("Node: analyze_image_node")
    image_path = state['image_path']
    query = state['current_query']
    if not image_path:
        return {"image_analysis": "Error: Could not find image to analyze."}
    try:
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        prompt_text = f"""You are a data analyst. Your task is to analyze the provided image to answer the user's query.
The user asked: '{query}'
Based *only* on the image, provide a 'Final Verdict' or 'Analysis' of the data.
Look for anomalies, skewdness, trends, or any key insights visible in the plot.
Be concise."""
        content = [{"type": "text", "text": prompt_text},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]
        msg = HumanMessage(content=content)
        response = vision_model.invoke([msg])
        print(f"Image Analysis: {response.content}")
        return {"image_analysis": response.content}
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return {"image_analysis": f"Error during image analysis: {e}"}


def final_response_node(state: AgentState) -> dict:
    """
    Node 7 (UPGRADED): Generates the final response.
    Handles EITHER a full report OR a simple chat history answer.
    """
    print("Node: final_response_node")

    # Check if we have SQL results. If not, we came from the intent router.
    sql_query = state.get('sql_query')

    if sql_query:
        # --- PATH 1: Full Report Generation ---
        print("Generating: Full Report")
        query = state['current_query']
        sql_result_str = state['sql_result_str']
        analysis_code = state.get('analysis_code')
        analysis_result = state.get('analysis_result')
        image_analysis = state.get('image_analysis')

        summary_prompt = f"""You are a helpful AI assistant.
Your job is to provide a final, natural language summary to the user.
Base your summary on the "Final Verdict" or "Statistical Result" if they exist, otherwise use the "Raw Data".

User's Original Question: {query}
--- CONTEXT ---
Raw Data:
{sql_result_str}
Statistical Result:
{analysis_result}
Final Verdict / Image Analysis:
{image_analysis}
--- END CONTEXT ---
Provide a concise, natural language answer to the user's original question: '{query}'"""

        summary_response = model.invoke(summary_prompt)
        summary = summary_response.content

        output_parts = [
            "Here is a complete summary of your request:",
            "\n### 💡 Final Summary\n",
            summary,
            "\n\n---",
            "\n### ⚙️ Execution Details\n"
        ]
        output_parts.append(f"**1. SQL Query:**\n```sql\n{sql_query}\n```\n")
        output_parts.append(f"**2. Raw Data Output:**\n```\n{sql_result_str}\n```\n")
        if analysis_code:
            output_parts.append(f"**3. Analysis Code:**\n```python\n{analysis_code}\n```\n")
        if analysis_result:
            output_parts.append(f"**4. Statistical Result:**\n```\n{analysis_result}\n```\n")
        if image_analysis:
            output_parts.append(f"**4. Image Analysis Verdict:**\n```\n{image_analysis}\n```\n")

        final_output = "".join(output_parts)

    else:
        # --- PATH 2: Chat History Answer ---
        print("Generating: Chat History Answer")
        messages = state['messages']
        current_query = messages[-1].content

        # Format history for the LLM
        history_str = "\n".join(
            [f"{m.type}: {m.content}" for m in messages]
        )

        prompt = f"""You are a helpful AI assistant.
Answer the "User's Latest Query" based *only* on the provided "Chat History".
Be conversational and concise.

Chat History:
{history_str}

User's Latest Query: {current_query}

Answer:"""

        response = model.invoke(prompt)
        final_output = response.content

    return {"messages": [AIMessage(content=final_output)]}


# --- 5. Build the Graph ---

print("Building LangGraph agent...")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify_intent", classify_intent_node)  # NEW
workflow.add_node("set_current_query", set_current_query_node)  # NEW
workflow.add_node("rag", rag_node)
workflow.add_node("generate_sql", sql_generation_node)
workflow.add_node("execute_sql", sql_execution_node)
workflow.add_node("generate_response", final_response_node)
workflow.add_node("generate_analysis_code", analysis_code_generation_node)
workflow.add_node("execute_analysis", execute_analysis_node)
workflow.add_node("analyze_image", analyze_image_node)

# --- Define Edges (UPDATED) ---

# 1. Set the new entry point
workflow.set_entry_point("classify_intent")

# 2. Add the new conditional entry router
workflow.add_conditional_edges(
    "classify_intent",
    classify_intent_node,  # The function that makes the decision
    {
        "answer_from_history": "generate_response",  # Route 1: Skip to the end
        "new_data_query": "set_current_query"  # Route 2: Start the full flow
    }
)

# 3. The "new data query" flow
workflow.add_edge("set_current_query", "rag")  # <-- Start of the original flow
workflow.add_edge("rag", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")

# 4. The first conditional router (decides if analysis is needed)
workflow.add_conditional_edges(
    "execute_sql",
    router_node,
    {
        "complex_analysis_branch": "generate_analysis_code",
        "final_response_branch": "generate_response"
    }
)

# 5. Analysis code execution flow
workflow.add_edge("generate_analysis_code", "execute_analysis")

# 6. The second conditional router (decides if image analysis is needed)
workflow.add_conditional_edges(
    "execute_analysis",
    analysis_router_node,
    {
        "analyze_image_branch": "analyze_image",
        "final_response_branch": "generate_response"
    }
)

# 7. Image analysis flows to final response
workflow.add_edge("analyze_image", "generate_response")

# 8. Final response node ends the graph
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()
print("Graph compiled. Agent is ready.")


# --- 6. Main Interaction Loop ---
def main():
    global data_engine, embedding_manager

    print("--- AI-Driven DB RAG & Analytics ---")

    # --- Step 1: Load Data ---
    print("\n📋 Supported formats:")
    print("  CSV:        /path/to/file.csv")
    print("  PostgreSQL: postgresql://user:pass@host:port/db")
    print("  SQLite:     sqlite:///path/to/database.db (e.g., 'sqlite:///example.db')")

    source = input("\n🔗 Enter data source (or 'exit'): ").strip()
    if source.lower() == 'exit':
        return

    print(f"\n🔄 Loading {source}...")
    if not data_engine.load(source):
        print("❌ Failed to load source. Exiting.")
        return

    print("✅ Data loaded successfully!")
    data_engine.show_database_structure()

    # --- Step 2: Build Vector Stores ---
    print("\n🧠 Building RAG vector stores for this data source...")
    embedding_manager = EmbeddingManager(data_engine)
    embedding_manager.build_vector_stores()
    print("✅ RAG system is online.")

    # --- Step 3: Chat Loop ---
    print("\n💬 Chat with your data! (Type 'exit' to quit)")

    # Initialize state with an empty message list
    current_state = {"messages": []}

    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            print("\n👋 Goodbye!")
            break

        # Add the new human message to the *current* state
        current_state["messages"].append(HumanMessage(content=query))

        # Invoke the graph with the *updated* state
        final_state = app.invoke(current_state)

        # The 'final_state' contains the *entire* history,
        # including the new AI response.
        ai_response = final_state['messages'][-1].content
        print(f"\nAI: {ai_response}")

        # The 'final_state' becomes the 'current_state' for the next loop,
        # preserving the chat history.
        current_state = final_state


if __name__ == "__main__":
    try:
        if not os.path.exists("example.db"):
            print("Creating dummy 'example.db' for first-time run...")
            from embedding_manager import main as create_dummy_db

            create_dummy_db()
            print("Dummy 'example.db' created.")
    except Exception as e:
        print(f"Could not create dummy db: {e}")

    main()