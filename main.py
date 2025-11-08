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
# --- UPDATED IMPORT ---
# We now import the GCS uploader here as well
from image_result import save_plot, save_to_excel, save_to_pdf
from upload_to_uri import upload_to_gcs

# Load environment variables (for Google API key)
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY not found. Please set it in .env file.")
    exit()

# --- 1. Setup Models, DataEngine, and EmbeddingManager ---

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# Use a vision-capable model
vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

data_engine = DataEngine()
embedding_manager = None  # Will be initialized after data is loaded


# --- 2. Define Agent State (UPDATED) ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    rag_context: str
    sql_query: str
    sql_result_str: str
    sql_result_df: Any  # The raw DataFrame for analysis
    current_query: str
    export_type: str  # "none", "pdf", "excel" - set at the start

    # --- Re-architected file/analysis fields ---
    analysis_code: str

    # This list will hold dicts of generated files
    # e.g., {"type": "plot", "caption": "...", "local_path": "...", "uri": "..."}
    generated_files: Annotated[list, operator.add]

    # This will hold the text analysis of *plots* only
    image_analysis_results: Annotated[list, operator.add]


# --- 3. Define Agent Nodes ---

def start_node(state: AgentState) -> dict:
    """ Node 0: The entry point. """
    print("Node: start_node")
    return {}  # Does not modify the state


def classify_intent_node(state: AgentState):
    """ Conditional routing function for the entry point. """
    print("Node: classify_intent_node (Routing)")
    messages = state['messages']

    if len(messages) <= 1:
        print("Routing: new_data_query (first message)")
        return "new_data_query"

    current_query = messages[-1].content
    history_str = "\n".join(
        [f"{m.type}: {m.content}" for m in messages[:-1]]
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
    Node 1.B (UPDATED): Sets current_query and checks for export request.
    """
    print("Node: set_current_query_node")
    last_message = state['messages'][-1]
    query = last_message.content

    # --- UPDATED EXPORT CHECK PROMPT ---
    export_prompt = f"""Analyze the user's query: "{query}"
Does the query ask to export, save, download, or create a report/file?
- If it mentions "excel", "csv", or "xlsx", respond with only the word: **excel**
- If it mentions "pdf", respond with only the word: **pdf**
- Otherwise, respond with only the word: **none**
"""
    export_response = model.invoke(export_prompt)
    export_type = export_response.content.strip().lower()

    print(f"Export flag set to: '{export_type}'")

    return {
        "current_query": query,
        "export_type": export_type,
        # --- NEW: Initialize lists ---
        "generated_files": [],
        "image_analysis_results": []
    }


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

CRITICAL RULES:
1.  If the user's query is vague and could apply to multiple tables (e.g., "show me the data", "export a table"), first try to find the *most relevant* table from the context.
2.  If it is *still* unclear, you MUST ask a clarifying question. To do this, generate the following SQL:
    `SELECT "AMBIGUOUS_QUERY" AS error, "Which table are you referring to?" AS clarification`
3.  The database is SQLite. Do not use functions like `DATABASE()`. For table counts, you can query `sqlite_master`.

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
    """
    Node 5: LLM-based router.
    Decides if the query needs analysis or just a simple answer.
    """
    print("Node: router_node (Routing)")
    if state['sql_result_df'] is None or state['sql_result_df'].empty:
        print("Routing: No data from SQL, skipping analysis.")
        return "final_response_branch"

    query = state['current_query'].lower()
    data_head = state['sql_result_df'].head().to_string()

    prompt = f"""You are a routing agent. Your job is to decide if a user's query requires complex analysis (like statistics or plotting) or if the raw data is a sufficient answer.
Do NOT consider export requests (like "save to excel"), only focus on analysis.

User Query: "{query}"
Data from SQL:
{data_head}

--- ROUTING RULES ---

1.  **Route: 'complex_analysis'**
    Choose this if the query asks for insights, plots, or calculations *about* the data.
    - "plot the..."
    - "visualize..."
    - "what is the correlation..."
    - "find anomalies in..."
    - "calculate the skewness of..."

2.  **Route: 'simple_response'**
    Choose this if the query just asks to *see* the data that was already fetched.
    - "show me all users"
    - "how many orders are there"
    - "list the products"
    - "export all users to excel" (This is simple_response, the export is handled elsewhere)

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
    """ Node 6.B (UPDATED): Executes analysis code, saves plot, uploads, and returns file info. """
    print("Node: execute_analysis_node")
    code = state['analysis_code']
    df = state['sql_result_df']
    query = state['current_query']

    exec_scope = {"plt": plt, "pd": pd, "df": df, "is_plot": None, "fig": None, "analysis_result": None}

    try:
        exec(code, exec_scope)
        is_plot = exec_scope.get('is_plot')

        if is_plot is True:
            fig = exec_scope.get('fig')
            if fig and isinstance(fig, plt.Figure):
                # Save plot locally and upload
                local_path, uri = save_plot(fig, query)
                return {"generated_files": [{
                    "type": "plot",
                    "caption": f"Visualization for: {query}",
                    "local_path": local_path,
                    "uri": uri
                }]}
            else:
                return {"generated_files": [{"type": "error", "caption": "Plot Generation Error",
                                             "result_text": "Code did not create a valid 'fig' object."}]}

        elif is_plot is False:
            result = exec_scope.get('analysis_result')
            print(f"Statistical Result: {result}")
            return {"generated_files": [{
                "type": "stat",
                "caption": f"Statistical result for: {query}",
                "result_text": str(result)
            }]}
        else:
            return {"generated_files": [{"type": "error", "caption": "Code Execution Error",
                                         "result_text": "Code did not set 'is_plot' boolean."}]}

    except Exception as e:
        print(f"❌ Error executing analysis code: {e}")
        return {"generated_files": [{"type": "error", "caption": "Code Execution Error", "result_text": str(e)}]}


def analysis_router_node(state: AgentState):
    """ Node 6.C (UPDATED): Checks output of previous step and routes. """
    print("Node: analysis_router_node (Routing)")

    last_file = state.get('generated_files', [{}])[-1]

    if last_file.get('type') == 'plot':
        print("Routing: analyze_image_branch")
        return "analyze_image_branch"
    else:
        # For 'stat' or 'error', go to the export check
        print("Routing: final_response_branch (to export check)")
        return "final_response_branch"


def analyze_image_node(state: AgentState) -> dict:
    """ Node 6.D (UPDATED): Sends the GCS URI to Gemini for analysis. """
    print("Node: analyze_image_node")

    last_file = state['generated_files'][-1]
    uri = last_file.get('uri')
    query = state['current_query']

    if not uri or "Error" in uri:
        print(f"❌ Skipping image analysis, URI not valid: {uri}")
        return {"image_analysis_results": ["Error: Could not upload image for analysis."]}

    try:
        prompt_text = f"""You are a data analyst. Your task is to analyze the provided image to answer the user's query.
The user asked: '{query}'
Based *only* on the image (which is at the public URL), provide a 'Final Verdict' or 'Analysis' of the data.
Look for anomalies, skewdness, trends, or any key insights visible in the plot.
Be concise."""

        # We provide the public GCS URL to the vision model
        content = [{"type": "text", "text": prompt_text},
                   {"type": "image_url", "image_url": {"url": uri}}]

        msg = HumanMessage(content=content)
        response = vision_model.invoke([msg])

        print(f"Image Analysis: {response.content}")
        return {"image_analysis_results": [response.content]}

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return {"image_analysis_results": [f"Error during image analysis: {e}"]}


def export_node(state: AgentState) -> dict:
    """
    Node 6.E (UPDATED): Saves the DataFrame, uploads it, and returns file info.
    """
    print("Node: export_node")
    export_type = state['export_type']
    query = state['current_query']
    df = state['sql_result_df']

    if df is None:
        return {"generated_files": [
            {"type": "error", "caption": "Export Error", "result_text": "No data available to export."}]}

    local_path, uri = None, None
    if export_type == "excel":
        local_path, uri = save_to_excel(df, query)
    elif export_type == "pdf":
        local_path, uri = save_to_pdf(df, query)

    if local_path and uri:
        return {"generated_files": [{
            "type": export_type,
            "caption": f"Data export for: {query}",
            "local_path": local_path,
            "uri": uri
        }]}
    else:
        return {"generated_files": [
            {"type": "error", "caption": "Export Error", "result_text": "File export or upload failed."}]}


def export_check_node(state: AgentState) -> dict:
    """
    Node 7: This is a simple passthrough node.
    """
    print("Node: export_check_node (Junction)")
    return {}


def export_check_router(state: AgentState):
    """
    Routing function for the export check.
    """
    print("Node: export_check_router (Routing)")
    if state['export_type'] in ("excel", "pdf"):
        if state['sql_result_df'] is not None:
            print("Routing: export_branch")
            return "export_branch"
        else:
            print("Routing: No data to export, skipping to response.")
            return "final_response_branch"
    else:
        print("Routing: No export requested, skipping to response.")
        return "final_response_branch"


def final_response_node(state: AgentState) -> dict:
    """
    Node 8 (UPDATED): Generates the final MARKDOWN response.
    """
    print("Node: final_response_node")
    sql_query = state.get('sql_query')

    if sql_query:
        # --- PATH 1: Full Report Generation ---
        print("Generating: Full Report (Markdown)")
        query = state['current_query']
        sql_result_str = state['sql_result_str']
        analysis_code = state.get('analysis_code')
        generated_files = state.get('generated_files', [])
        analysis_results = state.get('image_analysis_results', [])

        # --- Build the Markdown response ---
        output_parts = [f"Here is the result for your query: \"{query}\"\n"]

        # 1. Add generated files (plots, stats, exports)
        if generated_files:
            output_parts.append("## 💡 Analysis & Results\n")
            for f in generated_files:
                caption = f.get('caption', 'Result')
                f_type = f.get('type')
                uri = f.get('uri')

                if f_type == 'plot' and uri:
                    output_parts.append(f"**{caption}**\n")
                    output_parts.append(f"![{caption}]({uri})\n\n")
                elif f_type in ('excel', 'pdf') and uri:
                    output_parts.append(f"**{caption}**\n")
                    output_parts.append(f"[Click here to download your {f_type} report]({uri})\n\n")
                elif f_type == 'stat':
                    output_parts.append(f"**{caption}**:\n```\n{f.get('result_text')}\n```\n\n")
                elif f_type == 'error':
                    output_parts.append(f"**{caption}**:\n`{f.get('result_text')}`\n\n")

        # 2. Add visual analysis from the AI
        if analysis_results:
            output_parts.append("## 🤖 Visual Analysis Verdict\n")
            for res in analysis_results:
                output_parts.append(f"- {res}\n")

        # 3. Add execution details
        output_parts.append("\n---\n## ⚙️ Execution Details\n")
        output_parts.append(f"**SQL Query:**\n```sql\n{sql_query}\n```\n")
        output_parts.append(
            f"**Raw Data Output (first 10 rows):**\n```\n{state['sql_result_df'].head(10).to_string()}\n```\n")

        if analysis_code:
            output_parts.append(f"**Analysis Code:**\n```python\n{analysis_code}\n```\n")

        final_output = "".join(output_parts)

    else:
        # --- PATH 2: Chat History Answer ---
        print("Generating: Chat History Answer")
        messages = state['messages']
        current_query = messages[-1].content
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


# --- 5. Build the Graph (UPDATED) ---

print("Building LangGraph agent...")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("start_node", start_node)
workflow.add_node("set_current_query", set_current_query_node)
workflow.add_node("rag", rag_node)
workflow.add_node("generate_sql", sql_generation_node)
workflow.add_node("execute_sql", sql_execution_node)
workflow.add_node("generate_response", final_response_node)
workflow.add_node("generate_analysis_code", analysis_code_generation_node)
workflow.add_node("execute_analysis", execute_analysis_node)
workflow.add_node("analyze_image", analyze_image_node)
workflow.add_node("export_node", export_node)
workflow.add_node("export_check_node", export_check_node)

# --- Define Edges (UPDATED) ---

# 1. Set the new entry point
workflow.set_entry_point("start_node")

# 2. Add the new conditional entry router
workflow.add_conditional_edges(
    "start_node",
    classify_intent_node,
    {
        "answer_from_history": "generate_response",
        "new_data_query": "set_current_query"
    }
)

# 3. The "new data query" flow
workflow.add_edge("set_current_query", "rag")
workflow.add_edge("rag", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")

# 4. The main conditional router (decides analysis vs. simple)
workflow.add_conditional_edges(
    "execute_sql",
    router_node,
    {
        "complex_analysis_branch": "generate_analysis_code",
        "final_response_branch": "export_check_node"
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
        "final_response_branch": "export_check_node"
    }
)

# 7. Image analysis flows to the export check
workflow.add_edge("analyze_image", "export_check_node")

# 8. Add the new "export check" gate
workflow.add_conditional_edges(
    "export_check_node",
    export_check_router,
    {
        "export_branch": "export_node",
        "final_response_branch": "generate_response"
    }
)

# 9. Export node flows to final response
workflow.add_edge("export_node", "generate_response")

# 10. Final response node ends the graph
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()
app.get_graph().print_ascii()
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

    # ---

if __name__=="__main__":
    main()