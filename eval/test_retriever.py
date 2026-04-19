import pandas as pd
from src.srv.components import chroma_operator
from tqdm import tqdm
from src.srv.components.llm_operator import llm_request_normalizer

def open_test_data(path: str) -> pd.DataFrame:
    test_data = pd.read_csv(path)
    return test_data

def test_single_request(request: str, repo: str, right_chunk_id, k) -> int:
    docs = chroma_operator.get_top_k(request, repo=repo, k=k)
    doc_ids = [doc.id for doc in docs]
    return right_chunk_id in doc_ids
    
def test_retriever(vector_db_repo: str, path_to_test_data: str, k=3):
    test_data = open_test_data(path_to_test_data)
    total_test_samples = len(test_data)
    correct_accum = 0
    correct_accum_normalized = 0
    for _, test_sample in tqdm(test_data.iterrows()):
        correct_accum += test_single_request(test_sample["question"], repo=vector_db_repo, right_chunk_id=test_sample["chunk_id"], k=k)
        normalized_query = llm_request_normalizer.invoke({"user_query": test_sample["question"]}).content
        correct_accum_normalized += test_single_request(normalized_query, repo=vector_db_repo, right_chunk_id=test_sample["chunk_id"], k=k)
        
    print(f"Accuraccy without normalizing: {correct_accum / total_test_samples}")
    print(f"Accuraccy with normalizing: {correct_accum_normalized / total_test_samples}")
    
repo = "/home/infirriar/Documents/Projects/rag-code"
if __name__ == "__main__":
    test_retriever(vector_db_repo=repo, path_to_test_data="eval/dataset/test_data.csv", k=1)