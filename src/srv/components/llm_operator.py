from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from src.settings import settings

class LLMOperator:
    def __init__(self, file_name):
        self.model = ChatOllama(
            base_url=settings.instruct_llm_base_url,
            model=settings.instruct_llm_name
        )
        system_message = open(f"{settings.prompts_dir + file_name}--system.txt", "r").read()
        user_message = open(f"{settings.prompts_dir + file_name}--user.txt", "r").read()

        self.messages = ChatPromptTemplate(
            [
                ("system", system_message),
                ("user", user_message)
            ]
        )
        
    def stream(self, query, chunks: list[Document]):
        chunks = [chunk.page_content for chunk in chunks]
        fragments = "\n-----------------\n".join(chunks)
        print(fragments)
        chat = self.messages.invoke({"question": query, "fragments": fragments})
        for batch in self.model.stream(chat):
            yield batch.content
    
llm_operator = LLMOperator("give_advice")