from fastapi import FastAPI
from embs.encoders import bi_encoder
from loguru import logger
from pydantic import BaseModel

app = FastAPI()

class DocRequest(BaseModel):
    doc: str

class DocsRequest(BaseModel):
    docs: list

@app.post("/embedding/doc")
def doc_emb(request: DocRequest):
    logger.info(request.doc)
    res = bi_encoder.invoke(request.doc)
    res = res.detach().to("cpu").float().tolist()
    return res


@app.post("/embedding/docs")
def docs_emb(request: DocsRequest):
    res = bi_encoder.invoke(request.docs)
    res = res.detach().to("cpu").float().tolist()
    return res