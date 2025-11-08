# embedding_manager.py

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

from query_tool import DataEngine
from utils import get_embeddding


# 1. Create a custom embedding function for ChromaDB
#    This wrapper allows ChromaDB to use your 'get_embeddding' function
class GeminiEmbedder(embedding_functions.EmbeddingFunction):
    def __init__(self):
        # We don't need to pass the client here,
        # as utils.py handles the API key and client setup
        pass

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Our get_embeddding function is already batch-capable
        return get_embeddding(input)


# 2. Main Manager Class
class EmbeddingManager:
    def __init__(self, data_engine: DataEngine):
        print("Initializing EmbeddingManager...")
        self.client = chromadb.PersistentClient(path="./db_vector_stores")  # Stores on disk
        self.embedder = GeminiEmbedder()
        self.data_engine = data_engine

        # Create the two collections as requested
        self.schema_collection = self.client.get_or_create_collection(
            name="table_schemas",
            embedding_function=self.embedder
        )
        self.relation_collection = self.client.get_or_create_collection(
            name="relationships",
            embedding_function=self.embedder
        )
        print("ChromaDB collections 'table_schemas' and 'relationships' are ready.")

    def build_vector_stores(self):
        """
        Populates both vector stores based on the DataEngine's loaded data.
        This follows the logic from your diagram.
        """
        print("Building vector stores...")

        # --- 1. Process Table Schemas (as per diagram) ---
        print(f"Processing {len(self.data_engine.tables)} tables...")
        table_docs = []
        table_metadatas = []
        table_ids = []

        for i, table_name in enumerate(self.data_engine.tables):
            # Get schema
            schema_df = self.data_engine.get_schema(table_name)
            schema_str = schema_df.to_string()

            # Get 2-3 sample rows
            sample_rows_df = self.data_engine.query(f"SELECT * FROM {table_name} LIMIT 3")
            sample_rows_str = "No rows found."
            if sample_rows_df is not None and not sample_rows_df.empty:
                sample_rows_str = sample_rows_df.to_string()

            # Create the document to be embedded (as per diagram)
            doc_content = (
                f"Table Name: {table_name}\n\n"
                f"Schema:\n{schema_str}\n\n"
                f"Sample Rows:\n{sample_rows_str}"
            )

            table_docs.append(doc_content)
            table_metadatas.append({"table_name": table_name, "type": "schema"})
            table_ids.append(f"table_{i}")

        if table_docs:
            print(f"Adding {len(table_docs)} table schemas to vector store...")
            self.schema_collection.add(
                documents=table_docs,
                metadatas=table_metadatas,
                ids=table_ids
            )

        # --- 2. Process Relationships (as per diagram) ---
        print(f"Processing {len(self.data_engine.relationships)} relationships...")
        relation_docs = []
        relation_metadatas = []
        relation_ids = []

        for i, rel in enumerate(self.data_engine.relationships):
            # Create one-liner summary (as per diagram)
            doc_content = (
                f"Relationship: A link exists from table '{rel['from_table']}' "
                f"to table '{rel['to_table']}'.\n"
                f"Details: The column '{rel['from_columns']}' in '{rel['from_table']}' "
                f"references '{rel['to_columns']}' in '{rel['to_table']}'.\n"
                f"Constraint name: {rel['constraint']}"
            )

            relation_docs.append(doc_content)
            relation_metadatas.append({
                "from_table": rel['from_table'],
                "to_table": rel['to_table'],
                "type": "relationship"
            })
            relation_ids.append(f"rel_{i}")

        if relation_docs:
            print(f"Adding {len(relation_docs)} relationships to vector store...")
            self.relation_collection.add(
                documents=relation_docs,
                metadatas=relation_metadatas,
                ids=relation_ids
            )

        print("Vector stores built successfully.")

    def query_rag_context(self, query: str, n_results: int = 3) -> str:
        """
        Queries both collections and combines the results into a
        single context string to be passed to the LLM.
        """
        print(f"Querying RAG context for: '{query}'")

        # Query both collections
        schema_results = self.schema_collection.query(
            query_texts=[query],
            n_results=n_results
        )

        relation_results = self.relation_collection.query(
            query_texts=[query],
            n_results=n_results
        )

        context_parts = []

        # Add schema results
        context_parts.append("--- Relevant Table Schemas ---")
        if schema_results['documents']:
            for doc in schema_results['documents'][0]:
                context_parts.append(doc)
        else:
            context_parts.append("No relevant table schemas found.")

        # Add relationship results
        context_parts.append("\n--- Relevant Relationships ---")
        if relation_results['documents']:
            for doc in relation_results['documents'][0]:
                context_parts.append(doc)
        else:
            context_parts.append("No relevant relationships found.")

        return "\n".join(context_parts)


# --- Example of how to use this file ---
def main():
    # Example: Load a local SQLite DB
    db_path = "sqlite:///example.db"  # Create a dummy db for testing

    # Setup dummy data
    try:
        engine = DataEngine()
        engine.load(db_path)
        engine.query("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name VARCHAR, age INTEGER)")
        engine.query(
            "CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, item VARCHAR, FOREIGN KEY(user_id) REFERENCES users(id))")
        engine.query("INSERT INTO users (name, age) VALUES ('Avneesh', 30), ('Jane Doe', 25)")
        engine.query("INSERT INTO orders (user_id, item) VALUES (1, 'Laptop'), (2, 'Book')")
        print("Dummy database created and populated.")
    except Exception as e:
        print(f"Error setting up dummy db: {e}")
        return

    # 1. Connect to the data
    data_engine = DataEngine()
    if not data_engine.load(db_path):
        print("Failed to load data.")
        return

    print("DataEngine loaded.")
    data_engine.show_database_structure()

    # 2. Initialize the manager and build the stores
    manager = EmbeddingManager(data_engine)
    manager.build_vector_stores()

    # 3. Test a query
    test_query = "How many users are there and what did they order?"
    context = manager.query_rag_context(test_query)

    print("\n--- TEST RAG CONTEXT ---")
    print(context)
    print("------------------------")


if __name__ == "__main__":
    main()