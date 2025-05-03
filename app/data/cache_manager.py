import aiosqlite
import json

DB_PATH = "cache.db"

# EMBEDDING CACHE
async def get_cached_embedding(entity_type: str, entity_value: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT embedding FROM embedding_cache WHERE entity_type=? AND entity_value=?", 
            (entity_type, entity_value)
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])  # convert back from string
        return None

async def cache_embedding(entity_type: str, entity_value: str, embedding: list):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO embedding_cache (entity_type, entity_value, embedding) VALUES (?, ?, ?)",
            (entity_type, entity_value, json.dumps(embedding))
        )
        await db.commit()

# CYPHER QUERY CACHE
async def get_cached_cypher(question: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT cypher_query FROM cypher_cache WHERE question=?", 
            (question,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

async def cache_cypher(question: str, cypher_query: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO cypher_cache (question, cypher_query) VALUES (?, ?)",
            (question, cypher_query)
        )
        await db.commit()