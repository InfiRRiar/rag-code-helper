from transformers import AutoModel, AutoTokenizer
from src.embs.config import config
from src.secrets import secrets

MAX_LENGTH = 512

class Encoder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=secrets.hf_token
        )
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            token=secrets.hf_token
        )
    def invoke(self, texts: list):
        tokenized_text = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        emb = self.encoder(**tokenized_text)
        return emb
    
bi_encoder = Encoder(model_name=config.bi_encoder_name)