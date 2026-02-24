from langchain_core.embeddings import Embeddings
import httpx
import asyncio
from src.settings import settings
import torch

class Embedder(Embeddings):
    def __init__(self):
        super().__init__()
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = httpx.post(
            url=f'{settings.base_embeddings_url}/embedding/docs', 
            timeout=60,
            json={
                "docs": texts
            }
        ).json()
        return res
    
    def embed_query(self, text: str) -> list[float]:
        res = httpx.post(
            url=f'{settings.base_embeddings_url}/embedding/doc', 
            json={
                "doc": text
            }
        ).json()[0]
        return res