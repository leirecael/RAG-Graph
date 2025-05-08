from diskcache import Cache
from llm.llm_client import get_embedding
from data.database_connector import get_similar_nodes_by_entity
import os

DB_DIR = "./.cache"
DB_PATH = os.path.join(DB_DIR, "cache.db")

cache = Cache(DB_PATH)

async def get_embedding_cached(entity_value):
    result = cache.get(entity_value)
    if result is not None:
        print("EMB IN CACHE")
        return result, 0
    else:
        print("GENERATING EMB")
        emb,cost = await get_embedding(entity_value)
        cache.set(entity_value,emb,3600*24*7)
        return emb, cost

async def get_similarity_cached(entity_type, entity_value, embedding):
    key = f"{entity_type}:{entity_value}"
    result = cache.get(key)
    if result is not None:
        print("SIM IN CACHE")
        return result
    else:
        print("GENERATING SIM")
        nodes = await get_similar_nodes_by_entity(entity_type,embedding)
        cache.set(key,nodes,3600*24*7)
        return nodes
