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
# [Your existing embedding_manager.py code (classes) goes here]
# ... (GeminiEmbedder class)
# ... (EmbeddingManager class)

# --- Example of how to use this file ---
def main():
    """
    An elaborate test case to build and test the embedding manager.
    This creates a 4-table schema for e-commerce.
    """
    db_path = "sqlite:///elaborate_test.db"  # Use a new DB file

    # Clean up old DB file if it exists, for a fresh start
    import os
    if os.path.exists("elaborate_test.db"):
        os.remove("elaborate_test.db")

    print("--- 1. Setting up elaborate test database ('elaborate_test.db') ---")

    # Setup dummy data
    try:
        engine = DataEngine()
        engine.load(db_path)

        # Table 1: customers
        engine.query("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            email VARCHAR(100) UNIQUE,
            join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Table 2: products
        engine.query("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            category VARCHAR(50),
            price DECIMAL(10, 2)
        )""")

        # Table 3: orders
        engine.query("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )""")

        # Table 4: order_items (the "join" table for many-to-many)
        engine.query("""
        CREATE TABLE IF NOT EXISTS order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )""")

        print("✅ Tables created.")

        # Insert sample data
        engine.query(
            "INSERT INTO customers (first_name, last_name, email) VALUES ('Avneesh', 'User', 'avneesh@example.com'), ('Jane', 'Doe', 'jane@example.com'), ('Bob', 'Smith', 'bob@example.com')")
        engine.query(
            "INSERT INTO products (name, category, price) VALUES ('Laptop', 'Electronics', 1200.00), ('Mouse', 'Electronics', 25.50), ('Coffee Mug', 'Homeware', 15.00), ('Python Book', 'Books', 45.75)")

        # Avneesh's order
        engine.query("INSERT INTO orders (customer_id, status) VALUES (1, 'Shipped')")  # Order 1
        engine.query("INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 1, 1)")  # Laptop
        engine.query("INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 2, 1)")  # Mouse

        # Jane's order
        engine.query("INSERT INTO orders (customer_id, status) VALUES (2, 'Pending')")  # Order 2
        engine.query("INSERT INTO order_items (order_id, product_id, quantity) VALUES (2, 4, 2)")  # 2x Python Book

        # Avneesh's second order
        engine.query("INSERT INTO orders (customer_id, status) VALUES (1, 'Delivered')")  # Order 3
        engine.query("INSERT INTO order_items (order_id, product_id, quantity) VALUES (3, 3, 1)")  # Coffee Mug

        # Bob has no orders

        print("✅ Dummy database created and populated.")

    except Exception as e:
        print(f"❌ Error setting up dummy db: {e}")
        return

    # --- 2. Load the DataEngine and display structure ---
    print("\n--- 2. Loading DataEngine and Inspecting Structure ---")
    data_engine = DataEngine()
    if not data_engine.load(db_path):
        print("❌ Failed to load data.")
        return

    print("✅ DataEngine loaded.")
    data_engine.show_database_structure()

    # --- 3. Build the Vector Stores ---
    print("\n--- 3. Initializing EmbeddingManager and Building Stores ---")
    manager = EmbeddingManager(data_engine)
    manager.build_vector_stores()

    # --- 4. Test RAG Context Retrieval ---
    print("\n--- 4. Testing RAG Context Retrieval ---")

    test_queries = [
        "How many total orders are there?",
        "Show me products and their prices.",
        "What did Avneesh buy?",
        "Calculate the total revenue from all shipped orders.",
        "Which customers have not placed any orders?"
    ]

    for i, test_query in enumerate(test_queries):
        print(f"\n--- TEST QUERY {i + 1} ---")
        print(f"Query: '{test_query}'")
        context = manager.query_rag_context(test_query)
        print("\n--- RAG Context Generated ---")
        print(context)
        print("-------------------------------")


if __name__ == "__main__":
    main()