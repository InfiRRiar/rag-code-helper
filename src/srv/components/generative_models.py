from src.settings import settings
from langchain_openai import ChatOpenAI

gpt_5 = ChatOpenAI(
    name="gpt-5",
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    temperature=0.5
)
