from transformers import AutoModel, AutoTokenizer
from src.settings import settings
from loguru import logger
import torch

MAX_LENGTH = 1024

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
        
    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]    
    
    def invoke(self, texts: list) -> torch.Tensor:
        tokenized_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        emb = self.encoder(**tokenized_texts)
        pooled = self._last_token_pool(emb.last_hidden_state, tokenized_texts["attention_mask"])
        return pooled

bi_encoder = Encoder(model_name=settings.bi_encoder_name)