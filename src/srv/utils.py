from src.srv.components import ChromaOperator
from langchain_text_splitters import TextSplitter
import os

def is_rag_file(file_path: str):
    if file_path.startswith(".") or "_" in file_path:
        return False
    if not any([file_path.endswith(ext) for ext in [".py"]]):
        return False
    if any([sp in file_path for sp in [".venv"]]):
        return False
    return True

def check_repository_input(path: str):
    if not os.path.isdir(path):
        return False
    return True
    
def parse_repositry(path: str, chroma_operator: ChromaOperator, splitter: TextSplitter) -> tuple[int, int]:
    total_files = 0
    total_chunks = 0
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            if not is_rag_file(full_path):
                continue
            file_content = open(full_path, "r").read()
            parts = splitter.split_text(file_content)
            parts = list(map(lambda x: "Candidate code snippet:\n" + x, parts))
            chroma_operator.set_repository(path)
            chroma_operator.add_items(parts)
            
            total_files += 1
            total_chunks += len(parts)
    return total_files, total_chunks