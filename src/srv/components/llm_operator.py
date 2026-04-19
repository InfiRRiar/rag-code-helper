from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from src.settings import settings
from src.srv.components.generative_models import gpt_5

class LLMOperator:
    def __init__(self, file_name, model: Runnable = None):
        self.model = model
        if model is None:
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
    
llm_advice_giver = LLMOperator("give_advice", model=gpt_5)
llm_request_normalizer = LLMOperator("normalize")
llm_request_translator = LLMOperator("translate", model=gpt_5)