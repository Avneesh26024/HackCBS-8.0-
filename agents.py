from query_tool import DataEngine
from embedding_manager import EmbeddingManager

# Load database
engine = DataEngine()
engine.load("postgresql://postgres:avneesh26@localhost:5432/postgres")

# Generate schema
manager = EmbeddingManager(engine)
# schema = manager.generate_structured_schema_json()

# Save to file
manager.save_structured_schema_to_file("database_schema.json")