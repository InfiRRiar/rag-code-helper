import argparse
import asyncio
from src.srv.components import ChromaOperator, embedder
import os
from src.srv.utils import is_rag_file
from langchain_text_splitters import RecursiveCharacterTextSplitter

chroma_operator = ChromaOperator(emb_fun=embedder)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

async def main(path: str):
    total_files = 0
    total_chunks = 0
    if not os.path.isdir(path):
        print("Please make sure the path you've typed is actually a directory!")
        return
    if not chroma_operator.if_repo_exists(path):
        print("The repository is not analyzed. Initiate the process...")
        for root, _, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                if not is_rag_file(full_path):
                    continue
                file_content = open(full_path, "r").read()
                parts = splitter.split_text(file_content)
                chroma_operator.set_repo(path)
                chroma_operator.add_items(parts)
                
                total_files += 1
                total_chunks += len(parts)
                
        print(f"{total_files} files processed. {total_chunks} ejected")


def set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", "-r", help="The absolute path to the repository you need help with")
    
    return parser

if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    asyncio.run(main(path=args.repository))
