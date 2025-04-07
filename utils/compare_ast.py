import ast
import difflib

def get_ast_tokens(code):
    """
    將 Python 原始碼轉換為 AST token 串列（忽略變數名與排版）
    """
    tree = ast.parse(code)
    tokens = []

    class ASTVisitor(ast.NodeVisitor):
        def generic_visit(self, node):
            tokens.append(type(node).__name__)
            super().generic_visit(node)

    ASTVisitor().visit(tree)
    return tokens

def calculate_similarity(tokens1, tokens2):
    """
    計算兩份 token 串列的相似度
    """
    sm = difflib.SequenceMatcher(None, tokens1, tokens2)
    return sm.ratio()

def compare_files(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f:
        code1 = f.read()
    with open(file2, 'r', encoding='utf-8') as f:
        code2 = f.read()

    tokens1 = get_ast_tokens(code1)
    tokens2 = get_ast_tokens(code2)

    similarity = calculate_similarity(tokens1, tokens2)
    print(f"🧠 AST Structural Similarity: {similarity * 100:.2f}%")

if __name__ == "__main__":
    # 你可以改成自己要比對的檔案
    compare_files("main.py", "main (1).py")