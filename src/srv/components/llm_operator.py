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
        
    def invoke(self, data: dict):
        chain = self.messages | self.model
        return chain.invoke(data)
        
    def stream(self, data: dict):
        chat = self.messages.invoke(data)
        for batch in self.model.stream(chat):
            yield batch.content
    
llm_advice_giver = LLMOperator("give_advice")
llm_request_normalizer = LLMOperator("normalize")