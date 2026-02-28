from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from src.settings import settings

class LLMOperator:
    def __init__(self):
        self.model = ChatOllama(
            base_url=settings.instruct_llm_base_url,
            model=settings.instruct_llm_name
        )
    
    def stream(self, query, chunks: list[Document]):
        messages = [HumanMessage(content=query)]
        for batch in self.model.stream(messages):
            yield batch.content
    
llm_operator = LLMOperator()