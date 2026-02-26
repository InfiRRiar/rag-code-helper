import argparse
import asyncio
from src.srv.components import ChromaOperator, embedder, ASTSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.srv.utils import check_repository_input, parse_repositry

chroma_operator = ChromaOperator(emb_fun=embedder)
# splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=1024, chunk_overlap=50, language=Language.PYTHON)
splitter = ASTSplitter(chunk_size=1024)

async def main_cycle(path: str):
    if not check_repository_input(path):
        print("Please make sure the path you've typed is actually a directory!")
    if not chroma_operator.if_repo_exists(path):
        print("The repository is not analyzed. Initiate the process...")
        total_files, total_chunks = parse_repositry(
            path=path,
            chroma_operator=chroma_operator,
            splitter=splitter
        )
        print(f"{total_files} files processed. {total_chunks} ejected")
        
    while True:
        print("> ", end="")
        question = input()
        query = "Find the most relevant code snippet given the following query:\n" + question
        chunks = chroma_operator.get_top_k(query=query, repo=path, k=3)
        
def set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", "-r", help="The absolute path to the repository you need help with")
    
    return parser

if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    asyncio.run(main_cycle(args.repository))
