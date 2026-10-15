# Modified By: Callam
# Project: Lotto Generator
# Purpose: Structural map of steps/ and config/ for the Stage 1 info layer
# Description:
#   - Finds get_data / add_data keys (pipeline. and self.pipeline.)
#   - Same basename keys optuna_bridge PIPE_TO_FILENAME expects
#   - Nested functions do not drop parent add_data tracking
#   - Files that define get_encoder_spec are tagged encoder
#   - output_key string inside that return dict is listed under outputs
#     (real key from the file — not a fake add_data)

import os                              # Paths and directory walk
import ast                             # Parse source into a tree without running it
import logging                         # Load / parse status
from typing import Dict, List, Any     # Type hints for summaries

logging.basicConfig(                   # Timestamped INFO logs
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PIPE_METHODS = ("add_data", "get_data")  # Only these calls count as pipeline I/O


class PipelineVisitor(ast.NodeVisitor):
    """Walk functions and record pipeline.get_data / add_data string keys."""

    def __init__(self):
        self.functions = []            # One dict per function / nested function
        self._stack = []               # Current function on top so nested defs keep parent

    def visit_FunctionDef(self, node):
        func = {                       # Record for this def
            "name": node.name,         # Function name
            "args": [arg.arg for arg in node.args.args],  # Parameter names
            "calls": [],               # Other call names (not pipe I/O)
            "pipeline_data": [],       # get_data(key) / add_data(key) strings
            "encoder_output_key": None,  # Filled if this def is get_encoder_spec
        }
        self.functions.append(func)    # Keep even nested defs as their own rows
        self._stack.append(func)       # Nested visits write into this func
        self.generic_visit(node)       # Walk body
        self._stack.pop()              # Leave this def

    visit_AsyncFunctionDef = visit_FunctionDef  # Same handling for async def

    def visit_Call(self, node):
        if not self._stack:            # Call outside any function — skip recording
            self.generic_visit(node)
            return

        current = self._stack[-1]      # Innermost function

        call_name = None
        if isinstance(node.func, ast.Name):
            call_name = node.func.id   # bare get_data(...)
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr # pipeline.get_data(...) or self.pipeline.add_data(...)

        if call_name:
            if call_name in PIPE_METHODS and node.args:
                first = node.args[0]   # First arg should be the key
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    entry = f"{call_name}({first.value})"  # e.g. add_data(clusters)
                    if entry not in current["pipeline_data"]:
                        current["pipeline_data"].append(entry)
                else:
                    if call_name not in current["pipeline_data"]:
                        current["pipeline_data"].append(call_name)  # key was not a string literal
            else:
                if call_name not in current["calls"]:
                    current["calls"].append(call_name)

        self.generic_visit(node)       # Nested calls inside this call

    def visit_Return(self, node):
        if not self._stack:            # Return at module level — ignore
            self.generic_visit(node)
            return
        current = self._stack[-1]      # Function that owns this return
        if current["name"] != "get_encoder_spec":
            self.generic_visit(node)   # Not the encoder contract — keep walking
            return
        val = node.value               # Expression being returned
        if isinstance(val, ast.Dict):  # return { ... }
            for k, v in zip(val.keys, val.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "output_key"
                    and isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                ):
                    current["encoder_output_key"] = v.value  # e.g. quantum_features
        self.generic_visit(node)       # Nested nodes in the return value


class PipelineAssessment:
    """Structural analysis for steps/ and config/. Basename keys stay stable."""

    def __init__(self, steps_path: str = None, config_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # adaptor/assessment.py -> repo root
        self.steps_path = steps_path or os.path.join(base_dir, "steps")
        self.config_path = config_path or os.path.join(base_dir, "config")
        self.files: Dict[str, str] = {}  # full_path -> source text

    def discover_all_files(self) -> Dict[str, str]:
        self.files.clear()             # Fresh map each discover
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]  # Try until one decodes

        for directory in (self.steps_path, self.config_path):
            if not os.path.exists(directory):
                continue               # Missing folder is not an error
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
                            break      # Stop encoding loop on first success
                        except UnicodeDecodeError:
                            continue

        logging.info(f"Total files loaded: {len(self.files)}")
        return self.files

    def list_all_files(self) -> List[str]:
        return list(self.files.keys())  # Full paths currently loaded

    def parse_file(self, full_path: str) -> Dict[str, Any]:
        source = self.files.get(full_path)
        if source is None:             # Not in the cache — try disk
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
            tree = ast.parse(source, filename=full_path)  # Syntax tree only — no import
            visitor = PipelineVisitor()
            visitor.visit(tree)
            return {"file_path": full_path, "functions": visitor.functions}
        except Exception as e:
            return {"error": str(e)}   # Bad syntax or visitor failure

    def parse_all_files(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for full_path in self.files:
            fname = os.path.basename(full_path)  # Key matches PIPE_TO_FILENAME basenames
            results[fname] = self.parse_file(full_path)
        return results

    def get_pipe_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        One pass: reads/outputs/data_movement all come from the AST visitor
        so they cannot disagree with each other.
        Encoder files: component_class + output_key from get_encoder_spec.
        """
        if not self.files:
            self.discover_all_files()  # Lazy load if nothing cached

        parsed = self.parse_all_files()
        summary = {}

        for full_path in self.files:
            fname = os.path.basename(full_path)
            data = parsed.get(fname, {})

            reads: List[str] = []      # Unique get_data keys
            outputs: List[str] = []    # Unique add_data keys + encoder output_key
            pipeline_moves: List[str] = []  # Full get_data(k) / add_data(k) strings

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

            func_names = [func.get("name") for func in data.get("functions", [])]
            is_encoder = "get_encoder_spec" in func_names  # Contract present in this file
            for func in data.get("functions", []):
                key = func.get("encoder_output_key")
                if key and key not in outputs:
                    outputs.append(key)  # Same string as in get_encoder_spec

            summary[fname] = {
                "file_path": full_path,
                "function_count": len(data.get("functions", [])),
                "reads": reads,
                "outputs": outputs,
                "data_movement": pipeline_moves,
                "component_class": "encoder" if is_encoder else "pipe",
                "encoder_contract": is_encoder,
            }

        return summary


if __name__ == "__main__":
    assessor = PipelineAssessment()    # Default steps/ and config/ under repo root
    assessor.discover_all_files()
    summary = assessor.get_pipe_summary()

    print("\n" + "=" * 65)
    print(" PIPELINE ASSESSMENT SUMMARY")
    print("=" * 65)
    for fname, data in summary.items():
        print(f"\n{fname}")
        print(f"  Functions      : {data['function_count']}")
        print(f"  Class          : {data['component_class']}")
        print(f"  Reads          : {data['reads']}")
        print(f"  Outputs        : {data['outputs']}")
        if data["data_movement"]:
            print(f"  Data Movement  : {data['data_movement']}")