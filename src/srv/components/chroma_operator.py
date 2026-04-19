from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.srv.components.embedder import embedder


def strs_to_docs(items: list[str], repo_name: str):
    docs = []
    for item in items:
        doc_item = Document(
            page_content=item,
            metadata={"repo": repo_name}
        )
        docs.append(doc_item)
    return docs
class ChromaOperator:
    def __init__(self, emb_fun):
        self.vector_store = Chroma(
            collection_name="code_fragments", 
            embedding_function=emb_fun,
            persist_directory="./chroma_langchain_db",
        )
    
    def if_repo_exists(self, repo: str):
        res = self.vector_store.get(where={"repo": repo}, limit=1)
        return bool(res.get("documents"))
    
    def get_top_k(self, query: str, repo: str, k: int = 10):
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter={"repo": repo}
        )
    def add_items(self, items: list[str], repo: str):
        docs = strs_to_docs(items=items, repo_name=repo)
        self.vector_store.add_documents(documents=docs)

chroma_operator = ChromaOperator(emb_fun=embedder)