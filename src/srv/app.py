import argparse
from src.srv.components import ChromaOperator, embedder, ASTSplitter
from src.srv.utils import check_repository_input, parse_repositry
from src.srv.components.llm_operator import llm_advice_giver, llm_request_normalizer
import time

chroma_operator = ChromaOperator(emb_fun=embedder)
splitter = ASTSplitter()

def main_cycle(path: str):
    if not check_repository_input(path):
        print("Please make sure the path you've typed is actually a directory!")
    if not chroma_operator.if_repo_exists(path):
        print("The repository is not analyzed. Initiate the process...")
        total_files, total_chunks = parse_repositry(
            path=path,
            chroma_operator=chroma_operator,
            splitter=splitter
        )
        print(f"{total_files} files processed. {total_chunks} chunks ejected")
        
    while True:
        print("> ", end="")
        question = input()
        
        normalized_query = llm_request_normalizer.invoke({"user_query": question}).content
        print(normalized_query)
        query = "Find the most relevant code snippet given the following query:\n" + normalized_query
        
        chunks = chroma_operator.get_top_k(query=query, repo=path, k=5)
        fragments = "\n-----------------\n".join([chunk.page_content for chunk in chunks])
        for batch in llm_advice_giver.stream(
            data={
                "question": question,
                "fragments": fragments
            }
        ):
            print(batch, end="", flush=True)
        print()
        
def set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", "-r", help="The absolute path to the repository you need help with")
    
    return parser

if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    main_cycle(args.repository)
