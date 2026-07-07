"""
Pipeline Assessment Module

Purpose:
    Provides structured understanding of the pipeline by analyzing
    files in steps/ and config/. This serves as the knowledge layer
    for the self-improving adaptor, enabling better linkage between
    ablation performance signals and actual code structure.
"""

import os
import ast
import logging
import pprint
import re
from collections import defaultdict
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PipelineAssessment:
    """
    Main knowledge base for the LottoPipeline.

    Purpose:
        Provides structured, queryable understanding of all files in
        steps/ and config/. This forms the core structural analysis layer
        for the self-improving adaptor.

    Responsibilities:
        - Discover and load relevant .py files
        - Extract what each file reads and writes (via pipeline.get_data / add_data)
        - Parse code structure using AST (functions, arguments, and key calls)
        - Provide a unified summary that can be used by Optuna and ablation logic

    Note:
        This class performs static analysis only. It can tell you what
        functions exist and what calls they make, but it cannot show
        the actual calculated outputs or runtime behaviour of those
        calculations. A separate simulation/execution layer will be
        required later to observe what each calculation actually produces.
    """

    def __init__(self, steps_path: str = None, config_path: str = None):
        """
        Initializes the assessment module and resolves paths relative
        to the project root (assumes this file lives inside the adaptor/ folder).
        """
        # Resolve paths relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.steps_path = steps_path or os.path.join(base_dir, "steps")
        self.config_path = config_path or os.path.join(base_dir, "config")

        self.files: Dict[str, str] = {}   # full_path -> file_content


class PipelineVisitor(ast.NodeVisitor):
    """
    Helper class for AST traversal.

    Purpose:
        Extracts structured information from pipeline files.
        Collects function definitions, their arguments, and relevant
        function calls (especially pipeline.add_data, pipeline.get_data,
        and key operations like KMeans, MinMaxScaler, etc.).
    """

    def __init__(self):
        self.functions = []
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Record a new function and its arguments."""
        self.current_function = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "calls": []
        }
        self.functions.append(self.current_function)
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        """Record important function calls made inside the current function."""
        if self.current_function is not None:
            call_name = None

            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr

            if call_name:
                self.current_function["calls"].append(call_name)

        self.generic_visit(node)

    def get_file_content(self, filename: str) -> str:
        """
        Returns the content of a file by partial name match.
        Returns empty string if not found.
        """
        for path, content in self.files.items():
            if filename in path:
                return content
        return ""

    def list_all_files(self) -> List[str]:
        """Returns a list of all discovered full file paths."""
        return list(self.files.keys())

    def get_pipe_outputs(self) -> Dict[str, List[str]]:
        """
        Scans all loaded files and extracts the keys that are written
        using pipeline.add_data(...).
        Returns a dictionary: filename -> list of output keys.
        """
        import re
        from collections import defaultdict

        outputs: Dict[str, List[str]] = defaultdict(list)
        pattern = r'pipeline\.add_data\s*\(\s*["\']([^"\']+)["\']'

        for full_path, content in self.files.items():
            # Get just the filename (e.g. "clustering.py")
            filename = os.path.basename(full_path)

            matches = re.findall(pattern, content)
            for key in matches:
                if key not in outputs[filename]:
                    outputs[filename].append(key)

        return dict(outputs)
    
    def get_pipe_reads(self) -> Dict[str, List[str]]:
        """
        Scans all loaded files and extracts the keys that are read
        using pipeline.get_data(...).
        Returns a dictionary: filename -> list of read keys.
        """
        import re
        from collections import defaultdict

        reads: Dict[str, List[str]] = defaultdict(list)
        pattern = r'pipeline\.get_data\s*\(\s*["\']([^"\']+)["\']'

        for full_path, content in self.files.items():
            filename = os.path.basename(full_path)

            matches = re.findall(pattern, content)
            for key in matches:
                if key not in reads[filename]:
                    reads[filename].append(key)

        return dict(reads)
    
    def parse_file(self, full_path: str) -> Dict[str, Any]:
        """Parses one file with AST and returns structured function info."""
        if not os.path.exists(full_path):
            return {"error": f"File not found: {full_path}"}

        try:
            with open(full_path, "r", encoding="utf-8-sig") as f:
                source = f.read()

            tree = ast.parse(source, filename=full_path)
            visitor = PipelineVisitor()
            visitor.visit(tree)

            return {
                "file_path": full_path,
                "functions": visitor.functions
            }

        except Exception as e:
            return {"error": str(e)}
        
    def parse_all_files(self) -> Dict[str, Dict[str, Any]]:
        """Parses every discovered file and returns structured results."""
        results = {}

        for full_path in self.files.keys():
            filename = os.path.basename(full_path)
            results[filename] = self.parse_file(full_path)

        return results
    
    def get_main_functions(self) -> Dict[str, List[str]]:
        """
        Extracts the main function names defined in each file.
        Returns: filename -> list of function names
        """
        import re
        from collections import defaultdict

        functions: Dict[str, List[str]] = defaultdict(list)
        # Pattern to find function definitions
        pattern = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('

        for full_path, content in self.files.items():
            filename = os.path.basename(full_path)
            matches = re.findall(pattern, content, re.MULTILINE)

            for func_name in matches:
                if func_name not in functions[filename]:
                    functions[filename].append(func_name)

        return dict(functions)
    
    def get_pipe_summary(self) -> Dict[str, Dict[str, Any]]:
        """Main public method. Returns rich structured data for every file."""
        outputs = self.get_pipe_outputs()
        reads = self.get_pipe_reads()
        parsed = self.parse_all_files()

        summary = {}
        for full_path in self.files.keys():
            filename = os.path.basename(full_path)
            file_data = parsed.get(filename, {})

            summary[filename] = {
                "file_path": full_path,
                "functions": file_data.get("functions", []),
                "reads": reads.get(filename, []),
                "outputs": outputs.get(filename, []),
            }
        return summary
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Returns full structured info for one specific file."""
        for full_path in self.files.keys():
            if filename in full_path:
                parsed = self.parse_file(full_path)
                return {
                    "file_path": full_path,
                    "functions": parsed.get("functions", []),
                    "reads": self.get_pipe_reads().get(os.path.basename(full_path), []),
                    "outputs": self.get_pipe_outputs().get(os.path.basename(full_path), []),
                }
        return {"error": f"File not found: {filename}"}


if __name__ == "__main__":
    assessor = PipelineAssessment()
    files = assessor.discover_all_files()

    print("\n=== Discovered Files ===")
    for path in assessor.list_all_files():
        print(f"  - {path}")

    print(f"\nTotal files loaded: {len(files)}")
    print("\n=== Pipe Outputs ===")
    pprint.pprint(assessor.get_pipe_outputs())
    print("\n=== Main Functions ===")
    pprint.pprint(assessor.get_main_functions())
    print("\n=== Pipe Summary (Selected Files) ===")
    summary = assessor.get_pipe_summary()
    # Print only the most important files for now
    important_files = [
        "clustering.py",
        "bayesian_fusion.py",
        "deep_learning.py",
        "monte_carlo.py",
        "redundancy.py"
    ]



    for name in important_files:
        if name in summary:
            print(f"\n{name}:")
            pprint.pprint(summary[name])
    print("\n=== Pipe Reads (Test) ===")
    reads = assessor.get_pipe_reads()
    pprint.pprint(reads)

    print("\n=== Test parse_file (clustering.py) ===")
    result = assessor.parse_file("C:\\Users\\cjack\\Documents\\LottoPipeline\\steps\\clustering.py")
    pprint.pprint(result)
    print("\n=== Parse All Files (Summary) ===")
all_parsed = assessor.parse_all_files()

for filename, data in all_parsed.items():
    if "error" in data:
        print(f"{filename}: ERROR - {data['error']}")
    else:
        func_names = [f["name"] for f in data.get("functions", [])]
        print(f"{filename}: {len(func_names)} functions - {func_names}")