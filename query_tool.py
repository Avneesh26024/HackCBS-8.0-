# query_tool.py

import pandas as pd
import duckdb
from sqlalchemy import create_engine, inspect, text, MetaData, Table
from urllib.parse import quote_plus
import re
from tabulate import tabulate


class DataEngine:
    def __init__(self):
        self.duck = None
        self.engine = None
        self.source_type = None
        self.source = None
        self.tables = []
        self.inspector = None
        self.relationships = []

    # ----------------------------------------------------------
    # ✅ 1. CSV LOADER
    # ----------------------------------------------------------
    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            self.duck = duckdb.connect()
            self.duck.register("data", df)
            self.source_type = "csv"
            self.tables = ["data"]
            return True
        except Exception as e:
            print("CSV Load Error:", e)
            return False

    # ----------------------------------------------------------
    # ✅ 2. EXCEL LOADER
    # ----------------------------------------------------------
    def load_excel(self, path):
        try:
            df = pd.read_excel(path)
            self.duck = duckdb.connect()
            self.duck.register("data", df)
            self.source_type = "excel"
            self.tables = ["data"]
            return True
        except Exception as e:
            print("Excel Load Error:", e)
            return False

    # ----------------------------------------------------------
    # ✅ 3. SQL DATABASE LOADER (POSTGRES / MYSQL / SQLITE / SUPABASE)
    # ----------------------------------------------------------
    def load_sql(self, uri):
        try:
            # If user passed mysql:// without driver, prefer pymysql
            if uri.startswith("mysql://") and not uri.startswith("mysql+pymysql://"):
                uri = uri.replace("mysql://", "mysql+pymysql://", 1)

            # Supabase requires SSL
            if "supabase.co" in uri:
                self.engine = create_engine(uri, connect_args={"sslmode": "require"})
            else:
                self.engine = create_engine(uri)

            self.duck = duckdb.connect()
            self.inspector = inspect(self.engine)
            self.tables = self.inspector.get_table_names()

            # Load each SQL table → DuckDB
            for table in self.tables:
                try:
                    df = pd.read_sql(f"SELECT * FROM {table}", self.engine)
                    self.duck.register(table, df)
                    print(f"✓ Loaded table: {table} ({len(df)} rows)")
                except Exception as e:
                    print(f"⚠ Skipped table {table}: {e}")

            # Extract relationships
            self._extract_relationships()

            self.source_type = "sql"
            self.source = uri
            return True

        except Exception as e:
            print("SQL Load Error:", e)
            print("Check your credentials / SSL / Internet / Firewall")
            return False

    # ----------------------------------------------------------
    # ✅ 4. EXTRACT TABLE RELATIONSHIPS (FOREIGN KEYS)
    # ----------------------------------------------------------
    def _extract_relationships(self):
        """Extract foreign key relationships from the database."""
        try:
            self.relationships = []
            for table in self.tables:
                fks = self.inspector.get_foreign_keys(table)
                for fk in fks:
                    constrained_cols = ", ".join(fk["constrained_columns"])
                    referred_table = fk["referred_table"]
                    referred_cols = ", ".join(fk["referred_columns"])
                    self.relationships.append({
                        "from_table": table,
                        "from_columns": constrained_cols,
                        "to_table": referred_table,
                        "to_columns": referred_cols,
                        "constraint": fk.get("name", "fk_unknown")
                    })
        except Exception as e:
            print(f"⚠ Could not extract relationships: {e}")

    # ----------------------------------------------------------
    # ✅ 5. AUTOMATIC LOADER
    # ----------------------------------------------------------
    def load(self, source):
        """Detect type and load."""
        source = source.strip()

        if source.endswith(".csv"):
            return self.load_csv(source)

        if source.endswith(".xlsx") or source.endswith(".xls"):
            return self.load_excel(source)

        if source.startswith("postgresql://"):
            return self.load_sql(source)

        if source.startswith("mysql://") or source.startswith("mysql+pymysql://"):
            return self.load_sql(source)

        if source.startswith("sqlite:///"):
            return self.load_sql(source)

        print("Unsupported source format:", source)
        return False

    # ----------------------------------------------------------
    # ✅ 6. LIST ALL TABLES
    # ----------------------------------------------------------
    def get_tables(self):
        return self.tables

    # ----------------------------------------------------------
    # ✅ 7. GET SCHEMA FOR ANY TABLE (WITH PRIMARY/FOREIGN KEY INFO)
    # ----------------------------------------------------------
    def get_schema(self, table):
        try:
            if self.source_type == "sql":
                # Get columns
                columns = self.inspector.get_columns(table)
                schema_data = []

                # Get primary keys
                pk = self.inspector.get_pk_constraint(table)
                pk_cols = set(pk.get("constrained_columns", []))

                # Get foreign keys
                fk = self.inspector.get_foreign_keys(table)
                fk_dict = {}
                for f in fk:
                    for col in f["constrained_columns"]:
                        fk_dict[col] = f"FK → {f['referred_table']}({', '.join(f['referred_columns'])})"

                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"])
                    is_pk = "🔑 PK" if col_name in pk_cols else ""
                    is_fk = fk_dict.get(col_name, "")
                    nullable = "NULL" if col["nullable"] else "NOT NULL"

                    schema_data.append({
                        "Column": col_name,
                        "Type": col_type,
                        "Constraint": f"{is_pk} {is_fk} {nullable}".strip()
                    })

                return pd.DataFrame(schema_data)
            else:
                # For CSV / Excel → DuckDB inference
                return self.duck.execute(f"DESCRIBE {table}").fetchdf()

        except Exception as e:
            print("Schema Fetch Error:", e)
            return None

    # ----------------------------------------------------------
    # ✅ 8. SHOW TABLE RELATIONSHIPS / DIAGRAM
    # ----------------------------------------------------------
    def show_relationships(self):
        """Display foreign key relationships as a table."""
        if not self.relationships:
            print("ℹ No relationships found.")
            return

        print("\n" + "=" * 80)
        print("🔗 TABLE RELATIONSHIPS (Foreign Keys)")
        print("=" * 80)

        rel_data = []
        for rel in self.relationships:
            rel_data.append([
                rel["from_table"],
                rel["from_columns"],
                "→",
                rel["to_table"],
                rel["to_columns"]
            ])

        print(tabulate(rel_data, headers=["From Table", "Column(s)", "", "To Table", "Column(s)"], tablefmt="grid"))

        # ASCII diagram
        print("\n📊 Relationship Diagram:")
        for rel in self.relationships:
            print(f"  {rel['from_table']}.{rel['from_columns']} ──→ {rel['to_table']}.{rel['to_columns']}")

    # ----------------------------------------------------------
    # ✅ 9. SHOW DETAILED DATABASE STRUCTURE
    # ----------------------------------------------------------
    def show_database_structure(self):
        """Show all tables with row counts and relationships."""
        print("\n" + "=" * 80)
        print("📊 DATABASE STRUCTURE")
        print("=" * 80)

        table_info = []
        for table in self.tables:
            if self.source_type == "sql":
                row_count = int(pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", self.engine).iloc[0, 0])
                col_count = len(self.inspector.get_columns(table))
                fk_count = len(self.inspector.get_foreign_keys(table))
            else:
                df = self.duck.execute(f"SELECT * FROM {table}").fetchdf()
                row_count = len(df)
                col_count = len(df.columns)
                fk_count = 0

            table_info.append([table, col_count, row_count, fk_count])

        print(tabulate(table_info, headers=["Table", "Columns", "Rows", "Foreign Keys"], tablefmt="grid"))

        # Show relationships if any
        self.show_relationships()

    # ----------------------------------------------------------
    # ✅ 10. RUN QUERY
    # ----------------------------------------------------------
    def query(self, sql_query):
        """Execute a query. For SELECTs return dataframe. For DDL/DML, prefer running on original engine when available.

        Also normalizes a few Oracle-specific types (VARCHAR2, NUMBER) to be DuckDB/Postgres compatible
        and fixes common CREATE TABLE trailing-comma issues.
        """
        try:
            q = sql_query.strip()

            # Normalize Oracle-ish types to standard SQL types for execution in DuckDB
            def normalize(sql: str) -> str:
                s = sql
                # Common Oracle -> ANSI replacements
                s = re.sub(r"\bVARCHAR2\b", "VARCHAR", s, flags=re.IGNORECASE)
                s = re.sub(r"\bNVARCHAR2\b", "VARCHAR", s, flags=re.IGNORECASE)
                s = re.sub(r"\bNUMBER\b", "NUMERIC", s, flags=re.IGNORECASE)
                s = re.sub(r"\bCLOB\b", "TEXT", s, flags=re.IGNORECASE)
                s = re.sub(r"\bDATE\b", "TIMESTAMP", s, flags=re.IGNORECASE)
                # Remove trailing commas before closing paren in CREATE TABLE
                s = re.sub(r",\s*\)", r")", s)
                return s

            # If the user connected to an external SQL DB, for modifying queries run against that DB
            is_write = re.match(r"^(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|TRUNCATE|REPLACE)\b", q,
                                flags=re.IGNORECASE) is not None

            if is_write and self.source_type == "sql" and self.engine is not None:
                # Run on the original engine (SQLAlchemy) for DDL/DML
                try:
                    with self.engine.connect() as conn:
                        conn.execution_options(autocommit=True)
                        conn.execute(text(sql_query))
                    print("✓ Query executed on source DB.")
                    return pd.DataFrame()
                except Exception as e:
                    print("Source DB execution error:", e)
                    # fall back to trying in DuckDB after normalization

            # For SELECTs (or fallback), normalize and run in DuckDB
            normalized = normalize(sql_query)
            res = self.duck.execute(normalized)
            # DuckDB returns None for statements that don't return rows
            try:
                df = res.fetchdf()
                return df
            except Exception:
                # no rows to fetch
                print("✓ Query executed (no rows returned).")
                return pd.DataFrame()

        except Exception as e:
            print("Query Error:", e)
            return None

    # ----------------------------------------------------------
    # ✅ NEW: GET DATABASE STRUCTURE AS DICT
    # THIS IS THE CORRECTED FUNCTION
    # ----------------------------------------------------------
    def get_database_structure_dict(self):
        """Show all tables with row counts and relationships."""
        try:
            structure = {
                "tables": [],
                "relationships": self.relationships
            }

            for table in self.tables:
                if self.source_type == "sql":
                    # --- THIS IS THE FIXED LINE ---
                    row_count = int(pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", self.engine).iloc[0, 0])
                    # --- END OF FIX ---

                    col_count = len(self.inspector.get_columns(table))
                    fk_count = len(self.inspector.get_foreign_keys(table))
                else:
                    df = self.duck.execute(f"SELECT * FROM {table}").fetchdf()
                    row_count = len(df)
                    col_count = len(df.columns)
                    fk_count = 0

                structure["tables"].append({
                    "name": table,
                    "columns": col_count,
                    "rows": row_count,
                    "foreign_keys": fk_count
                })

            return structure

        except Exception as e:
            print(f"Error getting database structure: {e}")
            return {"error": str(e)}


# --------------------------------------------------
# ✅ CLI INTERFACE
# --------------------------------------------------
def print_help():
    """Print help menu."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║           COMMAND REFERENCE                                   ║
╚════════════════════════════════════════════════════════════════╝
  help           - Show this help menu
  schema [table] - Show detailed schema of a table
  relations      - Show all table relationships & foreign keys
  structure      - Show database structure overview
  tables         - List all available tables
  back           - Load a different data source
  exit           - Quit the application

  Or type any SQL query directly (SELECT, INSERT, CREATE, etc)
""")


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║        🔥 Combined Data Query Wrapper v2                      ║
║   Supports: CSV, XLSX, PostgreSQL, MySQL, SQLite              ║
║        with Table Relationships & Schema Analysis             ║
╚════════════════════════════════════════════════════════════════╝
""")

    engine = DataEngine()

    while True:
        print("\n📋 Supported formats:")
        print("  CSV:        /path/to/file.csv")
        print("  XLSX:       /path/to/file.xlsx")
        print("  PostgreSQL: postgresql://user:pass@host:port/db")
        print("  MySQL:      mysql+pymysql://user:pass@host:port/db")
        print("  SQLite:     sqlite:///path/to/database.db")

        source = input("\n🔗 Enter source (or 'exit' to quit): ").strip()

        if source.lower() == 'exit':
            print("\n👋 Goodbye!")
            break

        if not source:
            continue

        print(f"\n🔄 Loading {source}...")
        if engine.load(source):
            print("✅ Connected successfully!\n")

            # Show database structure
            engine.show_database_structure()

            # Show detailed schema for each table
            tables = engine.get_tables()
            print(f"\n{'=' * 80}")
            print("📋 DETAILED TABLE SCHEMAS")
            print(f"{'=' * 80}")

            for i, table in enumerate(tables, 1):
                print(f"\n🔍 TABLE {i}: {table}")
                print("-" * 50)
                schema = engine.get_schema(table)
                if schema is not None:
                    print(tabulate(schema, headers='keys', tablefmt='grid', showindex=False))
                else:
                    print("❌ Could not retrieve schema")

            # Query loop
            print(f"\n💻 Connected to: {engine.source}")
            print("💡 Type 'help' for available commands\n")

            while True:
                query = input("SQL> ").strip()

                if not query:
                    continue

                if query.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    return

                elif query.lower() == 'back':
                    print()
                    break

                elif query.lower() == 'help':
                    print_help()

                elif query.lower() == 'tables':
                    tables = engine.get_tables()
                    print(f"\n📊 Available tables ({len(tables)}):")
                    for t in tables:
                        print(f"  • {t}")

                elif query.lower().startswith('schema'):
                    parts = query.split()
                    if len(parts) > 1:
                        table = parts[1]
                        print(f"\n Schema for '{table}':")
                        schema = engine.get_schema(table)
                        if schema is not None:
                            print(tabulate(schema, headers='keys', tablefmt='grid', showindex=False))
                    else:
                        print("Usage: schema <table_name>")

                elif query.lower() in ['relations', 'relationships']:
                    engine.show_relationships()

                elif query.lower() == 'structure':
                    engine.show_database_structure()

                else:
                    # Execute as SQL query
                    result = engine.query(query)
                    if result is not None and not result.empty:
                        print("\n" + tabulate(result, headers='keys', tablefmt='grid', showindex=False))
                    elif result is not None:
                        pass  # Already printed success message
        else:
            print("❌ Failed to load source. Try again.")


if __name__ == "__main__":
    main()