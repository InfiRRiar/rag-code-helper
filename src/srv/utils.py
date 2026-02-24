def is_rag_file(file_path: str):
    if file_path.startswith(".") or "_" in file_path:
        return False
    if not any([file_path.endswith(ext) for ext in [".py"]]):
        return False
    if any([sp in file_path for sp in [".venv"]]):
        return False
    return True