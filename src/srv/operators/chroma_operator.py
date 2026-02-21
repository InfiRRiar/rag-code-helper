from langchain_chroma import Chroma
from langchain_core.runnables import chain
from langchain_core.documents import Document


class ChromaOperator:
    def __init__(self, emb_fun):
        self.vector_store = Chroma(
            collection_name="code_fragments", 
            embedding_function=emb_fun,
            persist_directory="./chroma_langchain_db"
        )
        self.repo_name = None
    
    def set_repo(self, repo_name: str):
        self.repo_name = repo_name
        
    def _strs_to_docs(self, items: list[str]):
        if self.repo_name is None:
            print("Please, set the repo first")
            return
        docs = []
        for item in items:
            doc_item = Document(
                page_content=item,
                metadata={"repo": self.repo_name}
            )
            docs.append(doc_item)
        return docs
    
    def get_top_k(self, query: str, repo: str, k: int = 10):
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter={"repo": repo}
        )
    
    async def add_items(self, items: list[str]):
        docs = self._strs_to_docs(items=items)
        await self.vector_store.aadd_documents(documents=docs)