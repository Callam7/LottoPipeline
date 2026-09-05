# Modified By: Callam
# Project: Lotto Generator
# Purpose: Structural map of steps/ and config/ for the Stage 1 info layer
# Description:
#   - Finds get_data / add_data keys (pipeline. and self.pipeline.)
#   - Same basename keys optuna_bridge PIPE_TO_FILENAME expects
#   - Nested functions do not drop parent add_data tracking

import os
import ast
import logging
from collections import defaultdict
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PIPE_METHODS = ("add_data", "get_data")


class PipelineVisitor(ast.NodeVisitor):
    """Walk functions and record pipeline.get_data / add_data string keys."""

    def __init__(self):
        self.functions = []
        self._stack = []

    def visit_FunctionDef(self, node):
        func = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "calls": [],
            "pipeline_data": [],
        }
        self.functions.append(func)
        self._stack.append(func)
        self.generic_visit(node)
        self._stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if not self._stack:
            self.generic_visit(node)
            return

        current = self._stack[-1]

        call_name = None
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr

        if call_name:
            if call_name in PIPE_METHODS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    entry = f"{call_name}({first.value})"
                    if entry not in current["pipeline_data"]:
                        current["pipeline_data"].append(entry)
                else:
                    if call_name not in current["pipeline_data"]:
                        current["pipeline_data"].append(call_name)
            else:
                if call_name not in current["calls"]:
                    current["calls"].append(call_name)

        self.generic_visit(node)


class PipelineAssessment:
    """Structural analysis for steps/ and config/. Basename keys stay stable."""

    def __init__(self, steps_path: str = None, config_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.steps_path = steps_path or os.path.join(base_dir, "steps")
        self.config_path = config_path or os.path.join(base_dir, "config")
        self.files: Dict[str, str] = {}

    def discover_all_files(self) -> Dict[str, str]:
        self.files.clear()
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

        for directory in (self.steps_path, self.config_path):
            if not os.path.exists(directory):
                continue
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if not filename.endswith(".py"):
                        continue
                    full_path = os.path.join(root, filename)
                    for enc in encodings:
                        try:
                            with open(full_path, "r", encoding=enc) as f:
                                self.files[full_path] = f.read()
                            logging.info(f"Loaded: {filename}")
                            break
                        except UnicodeDecodeError:
                            continue

        logging.info(f"Total files loaded: {len(self.files)}")
        return self.files

    def list_all_files(self) -> List[str]:
        return list(self.files.keys())

    def parse_file(self, full_path: str) -> Dict[str, Any]:
        source = self.files.get(full_path)
        if source is None:
            if not os.path.exists(full_path):
                return {"error": f"File not found: {full_path}"}
            encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
            for enc in encodings:
                try:
                    with open(full_path, "r", encoding=enc) as f:
                        source = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if source is None:
                return {"error": f"Could not decode: {os.path.basename(full_path)}"}

        try:
            tree = ast.parse(source, filename=full_path)
            visitor = PipelineVisitor()
            visitor.visit(tree)
            return {"file_path": full_path, "functions": visitor.functions}
        except Exception as e:
            return {"error": str(e)}

    def parse_all_files(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for full_path in self.files:
            fname = os.path.basename(full_path)
            results[fname] = self.parse_file(full_path)
        return results

    def get_pipe_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        One pass: reads/outputs/data_movement all come from the AST visitor
        so they cannot disagree with each other.
        """
        if not self.files:
            self.discover_all_files()

        parsed = self.parse_all_files()
        summary = {}

        for full_path in self.files:
            fname = os.path.basename(full_path)
            data = parsed.get(fname, {})

            reads: List[str] = []
            outputs: List[str] = []
            pipeline_moves: List[str] = []

            for func in data.get("functions", []):
                for item in func.get("pipeline_data", []):
                    if item not in pipeline_moves:
                        pipeline_moves.append(item)
                    if item.startswith("get_data(") and item.endswith(")"):
                        key = item[len("get_data("):-1]
                        if key and key not in reads:
                            reads.append(key)
                    elif item.startswith("add_data(") and item.endswith(")"):
                        key = item[len("add_data("):-1]
                        if key and key not in outputs:
                            outputs.append(key)

            summary[fname] = {
                "file_path": full_path,
                "function_count": len(data.get("functions", [])),
                "reads": reads,
                "outputs": outputs,
                "data_movement": pipeline_moves,
            }

        return summary


if __name__ == "__main__":
    assessor = PipelineAssessment()
    assessor.discover_all_files()
    summary = assessor.get_pipe_summary()

    print("\n" + "=" * 65)
    print(" PIPELINE ASSESSMENT SUMMARY")
    print("=" * 65)
    for fname, data in summary.items():
        print(f"\n{fname}")
        print(f"  Functions      : {data['function_count']}")
        print(f"  Reads          : {data['reads']}")
        print(f"  Outputs        : {data['outputs']}")
        if data["data_movement"]:
            print(f"  Data Movement  : {data['data_movement']}")