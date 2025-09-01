import os
import numpy as np
from sqlalchemy import create_engine, inspect, text
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

PG_CONN = os.getenv("PG_CONN", "postgresql://postgres:postgres@localhost:55432/northwind")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "northwind_rag"

# 1. Connect to Postgres
engine = create_engine(PG_CONN)
insp = inspect(engine)

# 2. Build table cards
def build_table_cards(engine, sample_rows=3):
    cards = []
    with engine.connect() as conn:
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            col_lines = [f"- {c['name']} ({c['type']})" for c in cols]
            fks = insp.get_foreign_keys(table)
            fk_lines = [f"- {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}" for fk in fks] or ["- none"]
            try:
                rows = conn.execute(text(f"SELECT * FROM {table} LIMIT {sample_rows}")).fetchall()
            except:
                rows = []
            card = f"""Table: {table}
Columns:
{chr(10).join(col_lines)}
Foreign Keys:
{chr(10).join(fk_lines)}
Row samples ({len(rows)}):
{chr(10).join([str(r) for r in rows])}
"""
            cards.append({"table": table, "text": card})
    return cards

cards = build_table_cards(engine)

# 3. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
embs = model.encode([c["text"] for c in cards], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

# 4. Push into Qdrant
client = QdrantClient(url=QDRANT_URL)

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=embs.shape[1], distance=Distance.COSINE)
)

points = [
    PointStruct(
        id=i,
        vector=emb.tolist(),
        payload={"table": c["table"], "text": c["text"], "type": "table_card"}
    )
    for i, (c, emb) in enumerate(zip(cards, embs))
]

client.upsert(collection_name=COLLECTION, points=points)

print(f"Uploaded {len(points)} table cards into Qdrant collection '{COLLECTION}'")
