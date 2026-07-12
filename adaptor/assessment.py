"""
Pipeline Assessment Module

Purpose:
    Provides clean, reliable structural understanding of all .py files 
    in steps/ and config/. This is the core knowledge layer for the adaptor.
"""

import os
import ast
import logging
import re
from collections import defaultdict
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PipelineVisitor(ast.NodeVisitor):
    """Extracts functions and pipeline data movement (add_data / get_data)."""

    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        func = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "calls": [],
            "pipeline_data": []
        }
        self.functions.append(func)
        self._current = func
        self.generic_visit(node)
        self._current = None

    def visit_Call(self, node):
        if not hasattr(self, "_current") or self._current is None:
            self.generic_visit(node)
            return

        call_name = None
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr

        if call_name:
            if call_name in ("add_data", "get_data") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    entry = f"{call_name}({first.value})"
                    if entry not in self._current["pipeline_data"]:
                        self._current["pipeline_data"].append(entry)
                else:
                    if call_name not in self._current["pipeline_data"]:
                        self._current["pipeline_data"].append(call_name)
            else:
                if call_name not in self._current["calls"]:
                    self._current["calls"].append(call_name)

        self.generic_visit(node)


class PipelineAssessment:
    """Main structural analysis module for the LottoPipeline."""

    def __init__(self, steps_path: str = None, config_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.steps_path = steps_path or os.path.join(base_dir, "steps")
        self.config_path = config_path or os.path.join(base_dir, "config")
        self.files: Dict[str, str] = {}

    def discover_all_files(self) -> Dict[str, str]:
        """Loads all .py files with robust multi-encoding support."""
        self.files.clear()
        directories = [self.steps_path, self.config_path]
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

        for directory in directories:
            if not os.path.exists(directory):
                continue
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.endswith(".py"):
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

    def get_pipe_outputs(self) -> Dict[str, List[str]]:
        outputs: Dict[str, List[str]] = defaultdict(list)
        pattern = r'pipeline\.add_data\s*\(\s*["\']([^"\']+)["\']'
        for full_path, content in self.files.items():
            fname = os.path.basename(full_path)
            for key in re.findall(pattern, content):
                if key not in outputs[fname]:
                    outputs[fname].append(key)
        return dict(outputs)

    def get_pipe_reads(self) -> Dict[str, List[str]]:
        reads: Dict[str, List[str]] = defaultdict(list)
        pattern = r'pipeline\.get_data\s*\(\s*["\']([^"\']+)["\']'
        for full_path, content in self.files.items():
            fname = os.path.basename(full_path)
            for key in re.findall(pattern, content):
                if key not in reads[fname]:
                    reads[fname].append(key)
        return dict(reads)

    def parse_file(self, full_path: str) -> Dict[str, Any]:
        if not os.path.exists(full_path):
            return {"error": f"File not found: {full_path}"}

        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
        source = None
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
        """Main public method. Returns clean structured data per file."""
        outputs = self.get_pipe_outputs()
        reads = self.get_pipe_reads()
        parsed = self.parse_all_files()

        summary = {}
        for full_path in self.files:
            fname = os.path.basename(full_path)
            data = parsed.get(fname, {})

            # Collect unique pipeline data movement
            pipeline_moves = []
            for func in data.get("functions", []):
                for item in func.get("pipeline_data", []):
                    if item not in pipeline_moves:
                        pipeline_moves.append(item)

            summary[fname] = {
                "file_path": full_path,
                "function_count": len(data.get("functions", [])),
                "reads": reads.get(fname, []),
                "outputs": outputs.get(fname, []),
                "data_movement": pipeline_moves,
            }
        return summary


# ===================== CLEAN TEST OUTPUT =====================

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
        if data['data_movement']:
            print(f"  Data Movement  : {data['data_movement']}")