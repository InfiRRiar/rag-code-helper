from langchain_ollama.chat_models import ChatOllama

model = ChatOllama(
    base_url="http://127.0.0.1:11434",
    model="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
    api_key="123",
)

for batch in model.stream("Generate 2 paragraphs thinking about current philosophical streams"):
    print(batch.content, end="", flush=True)