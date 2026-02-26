import ast
from langchain_text_splitters import TextSplitter


class ASTSplitter(TextSplitter):
    def __init__(self, chunk_size = 4000, chunk_overlap = 200, length_function = ..., keep_separator = False, add_start_index = False, strip_whitespace = True):
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator, add_start_index, strip_whitespace)
        
    def split_text(self, code: str) -> list[str]:
        funcs = list()
        ast_tree = ast.parse(code)
        for node in ast.walk(ast_tree):
            if any([isinstance(node, struct)] for struct in [ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]):
                segment = ast.get_source_segment(code, node)
                funcs.append(segment)
        print(funcs)