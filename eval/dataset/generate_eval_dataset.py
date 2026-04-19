import pandas as pd
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.srv.components import chroma_operator
from tqdm import tqdm
from src.settings import settings


prompt_template = ChatPromptTemplate(
    [
        ("system", "{system_prompt}"),
        ("user", "Code snippet:\n{code_snippet}")
    ]
)
model = ChatOpenAI(
    name="gpt-5",
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    temperature=0.5
)

chain = prompt_template | model | StrOutputParser()

with open("eval/prompt_templates/question_generate.txt") as f:
    prompt = f.read()

data = chroma_operator.vector_store._collection.get()

def generate_questions():
    test_questions = {
        "chunk_id": [],
        "question": []
    }
    for i in tqdm(range(len(data["ids"]))):
        content = data["documents"][i]
        doc_id = data["ids"][i]
        input = {
                "system_prompt": prompt,
                "code_snippet": content
            }
        res = chain.invoke(input).split("\n")
        for chunk in res:
            test_questions["chunk_id"].append(doc_id)
            test_questions["question"].append(chunk.strip())

    return test_questions

def save_reults(test_questions: dict, path: str):
    test_questions = pd.DataFrame(test_questions)
    test_questions = test_questions[test_questions["question"].str.len() > 0]
    test_questions.to_csv(path, index=False)

def main(path_to_save: str = "eval/dataset/test_data.csv"):
    questions_dict = generate_questions()
    save_reults(
        test_questions=questions_dict, 
        path=path_to_save
    )
    
if __name__ == "__main__":
    main()
