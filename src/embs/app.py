from fastapi import FastAPI
from src.embs.embedders import bi_encoder

app = FastAPI()

@app.get("/bi-encoder")
def get_bi_encoder_answer(texts: list):
    return bi_encoder.invoke(texts)