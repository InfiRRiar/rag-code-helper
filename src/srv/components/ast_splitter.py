import ast
from langchain_text_splitters import TextSplitter


class DefVisitor(ast.NodeVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_class = None
        self.found_nodes = []
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_function(self, node):
        extra_info = ""
        if self.current_class:
            extra_info += f"# class: {self.current_class}\n"
        self.found_nodes.append((extra_info, node))
        
    
    visit_FunctionDef = visit_function
    visit_AsyncFunctionDef = visit_function

class ASTSplitter(TextSplitter):
    def __init__(self, chunk_size = 4000, chunk_overlap = 200, length_function = ..., keep_separator = False, add_start_index = False, strip_whitespace = True):
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator, add_start_index, strip_whitespace)

    def split_text(self, code: str, file_location: str) -> list[str]:
        self.visitor = DefVisitor()
        chunks = list()
        
        ast_tree = ast.parse(code)
        self.visitor.visit(ast_tree)
        
        for extra_info, node in self.visitor.found_nodes:
            segment = ast.get_source_segment(code, node)
            extra_info = f"# full file path: {file_location}\n" + extra_info
            chunks.append(extra_info + segment)
        all_chunks = "\n\n\n".join(chunks)
        with open("logs.txt", "w") as file:
            file.write(all_chunks)
        return chunks