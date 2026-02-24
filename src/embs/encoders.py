from transformers import AutoModel, AutoTokenizer
from src.settings import settings
from loguru import logger
import torch

MAX_LENGTH = 512

class Encoder:
    def __init__(self, model_name: str):
        logger.info("Begin init")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=settings.hf_token
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            token=settings.hf_token
        )
        logger.info("init finished")
    def invoke(self, texts: list) -> torch.Tensor:
        tokenized_text = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        emb = self.encoder(**tokenized_text).last_hidden_state[:, 0, :]
        return emb
    
bi_encoder = Encoder(model_name=settings.bi_encoder_name)