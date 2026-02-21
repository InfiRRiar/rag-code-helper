import argparse
import asyncio
from src.srv.operators.chroma_operator import ChromaOperator

chroma_operator = ChromaOperator(emb_fun=lambda x: str(x))

async def main(path: str):
    print(path)

def set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", "-r", help="The absolute path to the repository you need help with")
    
    return parser

if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    asyncio.run(main(path=args.repository))
