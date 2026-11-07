# Modified By: Callam
# Project: Lotto Generator
# Purpose: Stage 2 brief — imprint from the info-layer decision
# Description:
#   - Reads adaptor_decision. Does not search. Does not call Ollama.
#   - Resolves the weak component to its source file.
#   - Parses that file with ast. Does not import it.
#   - Emits a scope dict: file, contract, 2–4 method queries, excerpt.
#   - Missing action, missing file, or no method signal returns None.
#   - Queries come from the method, never from the word lottery.

import ast                             # Parse the weak file without importing it
import logging                         # Same stamp format as the rest of the adaptor
import os                              # Path joins and existence checks
from typing import Any, Dict, List, Optional, Tuple  # Types for the scope dict

logging.basicConfig(                   # One log format for this module
    level=logging.INFO,                # INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Time + level + text
)

DECISION_KEY = "adaptor_decision"      # Pipeline key the info layer writes
REVIEW_ACTIONS = ("methodology_review", "encoder_review")  # Only these start a brief
MAX_QUERIES = 4                        # Cap. retrieve.py runs one search per query
MAX_EXCERPT_LINES = 80                 # Judge sees the writer, not the whole file
BANNED_QUERY_WORDS = ("lottery", "lotto", "powerball")  # Domain words never enter a query

PIPE_TO_FILENAME = {                   # Same map Stage 1 uses. Basename only.
    "bayesian_fusion_norm": "bayesian_fusion.py",  # Fusion step
    "monte_carlo": "monte_carlo.py",   # Monte Carlo step
    "redundancy": "redundancy.py",     # Sequential / redundancy step
    "markov_features": "markov.py",    # Markov step
    "entropy_features": "entropy.py",  # Entropy step
    "centroids": "clustering.py",      # Clustering file, centroid vector
    "clusters": "clustering.py",       # Same file, label vector
    "quantum_features": "quantum_features.py",  # Encoder module
    "quantum_kernels": "quantum_kernels.py",    # Encoder module
}

CONTRACT = {                           # Output key -> (width, kind). Judge must keep both.
    "bayesian_fusion_norm": (54, "max-scaled, not a simplex"),
    "monte_carlo": (54, "non-negative, sums to 1"),
    "redundancy": (54, "non-negative, sums to 1"),
    "markov_features": (54, "non-negative, sums to 1"),
    "entropy_features": (54, "non-negative, sums to 1"),
    "clusters": (54, "integer labels"),
    "centroids": (54, "float strengths in [0, 1]"),
    "quantum_features": (16, "encoder, width frozen"),
    "quantum_kernels": (24, "encoder, width frozen"),
}

QUERY_PHRASES = {                      # Candidate -> method phrases. No domain words.
    "clusters": [
        "k-means clustering alternatives",
        "gaussian mixture clustering",
        "density based clustering",
    ],
    "centroids": [
        "k-means clustering alternatives",
        "cluster centroid features",
        "gaussian mixture clustering",
    ],
    "monte_carlo": [
        "monte carlo sampling",
        "importance sampling",
        "weighted random sampling",
    ],
    "redundancy": [
        "recency features time series",
        "gap frequency features",
        "temporal redundancy features",
    ],
    "markov_features": [
        "markov chain cluster transitions",
        "first order markov features",
        "transition probability features",
    ],
    "entropy_features": [
        "shannon entropy features",
        "probability distribution entropy",
        "normalized entropy features",
    ],
    "bayesian_fusion_norm": [
        "bayesian fusion of probability vectors",
        "dirichlet posterior fusion",
        "log space probability fusion",
    ],
    "quantum_features": [
        "variational quantum feature map",
        "quantum expectation value features",
        "pennylane strongly entangling layers",
    ],
    "quantum_kernels": [
        "quantum fidelity kernel",
        "positive semi-definite kernel normalisation",
        "quantum kernel ridge features",
    ],
}

SKIP_MODULES = {                       # Imports that are not a method signal
    "os", "sys", "logging", "typing", "json", "math", "re", "datetime",
    "collections", "itertools", "functools", "pathlib", "sqlite3",
    "numpy", "np", "tensorflow", "keras", "sklearn",
}

DOC_URLS = {                           # Import prefix -> docs page for Exa find_similar
    "sklearn.cluster.KMeans": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
    "sklearn.cluster": "https://scikit-learn.org/stable/modules/clustering.html",
    "sklearn": "https://scikit-learn.org/stable/",
    "scipy.stats": "https://docs.scipy.org/doc/scipy/reference/stats.html",
    "scipy": "https://docs.scipy.org/doc/scipy/",
    "pennylane": "https://docs.pennylane.ai/en/stable/",
}


def _clean_query(text: str) -> str:
    """Lower, strip, and reject a phrase that names the lottery domain."""
    phrase = " ".join(text.lower().split())  # Collapse whitespace, lower case
    if not phrase:                     # Empty after clean
        return ""
    for banned in BANNED_QUERY_WORDS:  # Domain words are a bug, not a query
        if banned in phrase:
            return ""
    return phrase                      # Safe to send to OpenAlex / Exa / arXiv


def _resolve_file(decision: Dict[str, Any], repo_root: str) -> Optional[str]:
    """Prefer the assessment path. Fall back to the Stage 1 basename map."""
    structural = decision.get("structural") or {}  # Assessment row or empty
    reported = structural.get("file_path")  # Full path assessment stored
    if reported and os.path.isfile(reported):  # Path still exists on this machine
        return reported
    candidate = decision.get("candidate")  # Block name, e.g. clusters
    filename = PIPE_TO_FILENAME.get(candidate or "")  # Basename, or None
    if not filename:                   # Unknown candidate
        return None
    for folder in ("steps", "config"):  # Pipes live in steps, encoders in config
        path = os.path.join(repo_root, folder, filename)  # Candidate location
        if os.path.isfile(path):       # First hit wins
            return path
    return None                        # File not in either folder


def _read_source(path: str) -> Optional[str]:
    """Read the weak file. Try the encodings assessment.py already accepts."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):  # Same order as assessment
        try:
            with open(path, "r", encoding=encoding) as handle:  # Text mode
                return handle.read()  # Full source. Parsed, not imported.
        except UnicodeDecodeError:
            continue                   # Try the next encoding
    return None                        # Could not decode


def _collect_imports(tree: ast.AST) -> List[str]:
    """Top-level imports only. Skip stdlib and bare numpy / tensorflow."""
    found: List[str] = []              # Qualified names, insertion order
    for node in getattr(tree, "body", []):  # Module body. Nested imports stay out.
        if isinstance(node, ast.Import):  # import x, import y as z
            for alias in node.names:
                name = alias.name      # Dotted module name
                root = name.split(".")[0]  # First segment
                if root in SKIP_MODULES:  # Not a method signal
                    continue
                if name not in found:  # Keep first sighting
                    found.append(name)
        elif isinstance(node, ast.ImportFrom) and node.module:  # from x import Y
            root = node.module.split(".")[0]  # First segment of the source module
            if root in SKIP_MODULES and root != "sklearn":  # Keep sklearn. Names below.
                continue
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"  # sklearn.cluster.KMeans
                if qualified not in found:
                    found.append(qualified)
    return found                       # Method-bearing imports only


def _output_key(decision: Dict[str, Any]) -> Optional[str]:
    """The pipeline key this file must still emit."""
    candidate = decision.get("candidate")  # Block name is the key for every current pipe
    if candidate in CONTRACT:          # Known width and kind
        return candidate
    structural = decision.get("structural") or {}  # Assessment outputs, if present
    outputs = structural.get("outputs") or []  # add_data keys assessment found
    for key in outputs:                # Prefer a key we have a contract for
        if key in CONTRACT:
            return key
    return candidate                   # Unknown, but still name it


def _writer_excerpt(source: str, tree: ast.AST, output_key_name: str) -> str:
    """Function that add_data's the output key, capped. Fallback is the module head."""
    lines = source.splitlines()        # Index by AST lineno, which is 1-based
    target = None                      # FunctionDef that writes the key
    for node in ast.walk(tree):        # Nested defs included. Writer may be nested.
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):   # Look for add_data("key") in this function
            if not isinstance(child, ast.Call):
                continue
            func = child.func          # Call target
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "add_data" or not child.args:  # Not a pipeline write
                continue
            first = child.args[0]      # First arg should be the key
            if isinstance(first, ast.Constant) and first.value == output_key_name:
                target = node          # This function emits the contract key
                break
        if target is not None:         # Found the writer
            break
    if target is None:                 # No add_data of that key. Use the top of the file.
        return "\n".join(lines[:MAX_EXCERPT_LINES])
    start = max((target.lineno or 1) - 1, 0)  # 1-based lineno -> 0-based index
    end = target.end_lineno or (start + 1)  # End line, inclusive, 1-based
    snippet = lines[start:end]         # The writer function
    if len(snippet) > MAX_EXCERPT_LINES:  # Cap so the judge is not handed the whole file
        snippet = snippet[:MAX_EXCERPT_LINES]
    return "\n".join(snippet)          # Text excerpt, not an AST dump


def _queries_for(candidate: str, imports: List[str]) -> List[str]:
    """2–4 method phrases. Table first, then the imported class name."""
    phrases = list(QUERY_PHRASES.get(candidate, []))  # Known candidate
    if not phrases:                    # Unknown candidate: derive from imports
        for name in imports:
            leaf = name.split(".")[-1]  # KMeans, GaussianMixture
            if leaf[:1].isupper():     # Class-like name
                phrase = leaf.replace("_", " ")  # Split underscores only
                phrases.append(f"{phrase} alternatives")
                break
    cleaned: List[str] = []            # Deduped, banned-word-free
    for phrase in phrases:
        safe = _clean_query(phrase)    # Reject lottery wording
        if safe and safe not in cleaned:
            cleaned.append(safe)
        if len(cleaned) >= MAX_QUERIES:  # Cap
            break
    return cleaned                     # May be empty. Caller treats that as no brief.


def _similar_urls(imports: List[str]) -> List[str]:
    """Docs URLs for sklearn / scipy / pennylane imports. One per prefix."""
    urls: List[str] = []               # Insertion order, no duplicates
    for name in imports:
        for prefix, url in DOC_URLS.items():  # Longest prefix should win
            if name == prefix or name.startswith(prefix + "."):
                if url not in urls:
                    urls.append(url)
                break                  # First matching prefix is enough
    return urls                        # May be empty. Exa search still runs on queries.


def _contract_line(output_key_name: str) -> str:
    """One line the judge must keep: same add_data key, same width, same kind."""
    width, kind = CONTRACT.get(output_key_name, (None, "unknown kind"))  # Fallback
    width_text = str(width) if width is not None else "unknown width"
    return (
        f"must still add_data('{output_key_name}') "
        f"with shape ({width_text},) {kind}"
    )


def build_brief(
    decision: Optional[Dict[str, Any]],
    repo_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Turn an info-layer decision into a search scope.

    Returns None when Stage 2 should not start: no packet, wrong action,
    file not resolved, source unreadable, or no method query.
    """
    if not decision:                   # Info layer wrote nothing
        logging.info("Brief skipped: no decision packet.")
        return None
    action = decision.get("action")    # methodology_review / encoder_review / stack_review
    if action not in REVIEW_ACTIONS:   # stack_review has no candidate
        logging.info(f"Brief skipped: action={action}.")
        return None
    candidate = decision.get("candidate")  # Block name
    if not candidate:                  # Named action but no winner
        logging.info("Brief skipped: no candidate.")
        return None

    root = repo_root or os.getcwd()    # Caller should pass the repo root
    path = _resolve_file(decision, root)  # Assessment path, else steps/ or config/
    if path is None:                   # Cannot open a file we cannot find
        logging.info(f"Brief skipped: no file for candidate '{candidate}'.")
        return None
    source = _read_source(path)        # Text of the weak file
    if source is None:                 # Decode failed
        logging.info(f"Brief skipped: could not read {path}.")
        return None
    try:
        tree = ast.parse(source, filename=path)  # Syntax only. No import.
    except SyntaxError as exc:         # Bad file. Do not guess.
        logging.info(f"Brief skipped: {path} did not parse ({exc}).")
        return None

    key = _output_key(decision)        # Pipeline key the replacement must emit
    if key is None:                    # Should not happen after the candidate check
        logging.info("Brief skipped: no output key.")
        return None
    imports = _collect_imports(tree)   # Method-bearing imports
    queries = _queries_for(candidate, imports)  # 2–4 phrases, no lottery words
    if not queries:                    # Nothing for retrieve.py to search
        logging.info(f"Brief skipped: no queries for '{candidate}'.")
        return None

    width, kind = CONTRACT.get(key, (None, "unknown kind"))  # Contract table
    structural = decision.get("structural") or {}  # Reads, if assessment ran
    scope = {                          # Imprint. Later files do not reopen the weak file.
        "target_file": path,           # File the sketch would replace
        "candidate": candidate,        # Block name from the decision
        "candidate_class": decision.get("candidate_class"),  # pipe | encoder
        "output_key": key,             # add_data key that must survive
        "width": width,                # Vector width. None if unknown.
        "kind": kind,                  # integer labels, simplex, encoder, ...
        "reads": list(structural.get("reads") or []),  # Pipeline keys this file reads
        "imports": imports,            # Method signal from the file
        "similar_urls": _similar_urls(imports),  # Docs pages for Exa
        "queries": queries,            # What retrieve.py is allowed to search
        "excerpt": _writer_excerpt(source, tree, key),  # Writer function, capped
        "contract": _contract_line(key),  # One line for the judge
        "importance": decision.get("importance") or {},  # Scores. Context only.
    }
    logging.info(                       # One line so a test run shows the imprint
        f"Brief ready: {candidate} -> {os.path.basename(path)} | "
        f"key={key} width={width} queries={len(queries)}"
    )
    return scope                       # Ready for retrieve.py. No network was used.


if __name__ == "__main__":             # Local check. No search.
    fake = {                           # Stand-in for adaptor_decision
        "action": "methodology_review",
        "candidate": "clusters",
        "candidate_class": "pipe",
        "importance": {"clusters": -0.01},
        "structural": None,            # Force the basename fallback
    }
    root = os.getcwd()                 # Run this from the repo root
    result = build_brief(fake, root)   # Local imprint only
    if result is None:
        print("no brief")              # File missing, or queries empty
    else:
        print(result["target_file"])   # Resolved path
        print(result["output_key"], result["width"], result["kind"])
        print(result["queries"])       # 2–4 method phrases
        print(result["contract"])      # Line the judge must keep
        print("--- excerpt ---")
        print(result["excerpt"][:500])  # Head of the writer, so the test stays short
