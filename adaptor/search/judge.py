# Modified By: Callam
# Project: Lotto Generator
# Purpose: Stage 2 judge — rank methods that still fit the contract
# Description:
#   - Takes the brief scope and the pages extract already read.
#   - Sends both to the local Ollama model. One call.
#   - Keeps a row only if it has a URL that was in the page list
#     and it still emits the same key at the same width.
#   - No URL means reject. A wrong key or width means reject.
#   - Does not search. Does not open a project file. Does not edit.

import json                            # Parse the model reply
import logging                         # Same stamp format as the rest of the adaptor
from typing import Any, Dict, List     # Scope, pages, shortlist

import requests                        # Local Ollama HTTP. No Ollama package.

logging.basicConfig(                   # One log format for this module
    level=logging.INFO,                # INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Time + level + text
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"  # Local. No key.
JUDGE_MODEL = "qwen2.5:7b"             # Already installed. Not the coder model.
REQUEST_TIMEOUT = 180                  # Seconds. A 7b call is slower than a search.
SHORTLIST_CAP = 3                      # Stage 3 sees at most this many
PAGE_CHARS = 2500                      # Per page. The model does not need 8000.


class Judge:
    """
    Rank methods that still fit the brief contract.

    Input is the scope from build_brief and the pages from Extract.run.
    Output is a shortlist of dicts: method, url, sketch, rank.
    An empty list is no_viable. Nothing is written.
    """

    def run(self, scope: Dict[str, Any], pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One Ollama call. Drop anything that does not fit."""
        if not scope or not pages:     # Nothing to judge
            logging.info("Judge skipped: no scope or no pages.")
            return []
        allowed = {str(p.get("url") or "") for p in pages if p.get("url")}  # Only these URLs
        prompt = self._prompt(scope, pages)  # Contract plus the page text
        raw = self._ask(prompt)        # Local model
        if not raw:                    # Call failed or empty
            logging.info("Judge: no reply.")
            return []
        rows = self._parse(raw)        # JSON list, or empty
        kept: List[Dict[str, Any]] = []  # Rows that pass the gates
        for row in rows:               # Model order is the rank order
            url = str(row.get("url") or "")
            if url not in allowed:     # Invented URL, or no URL. Reject.
                logging.info(f"Judge dropped, URL not in pages: {url}")
                continue
            if not self._fits(row, scope):  # Wrong key or wrong width. Reject.
                logging.info(f"Judge dropped, contract miss: {url}")
                continue
            kept.append({
                "method": str(row.get("method") or "").strip(),  # Name of the method
                "url": url,            # Must be one we read
                "sketch": str(row.get("sketch") or "").strip(),  # How it sits in the file
                "rank": len(kept) + 1, # 1 is the best that survived
            })
            if len(kept) >= SHORTLIST_CAP:  # Cap for Stage 3
                break
        logging.info(f"Judge ready: {len(kept)} method(s) from {len(pages)} page(s).")
        return kept                    # [] is no_viable

    def _prompt(self, scope: Dict[str, Any], pages: List[Dict[str, Any]]) -> str:
        """Contract first, then each page, then the required JSON shape."""
        lines = [
            "You rank methods that can replace one block in a lottery pipeline.",
            "A method is usable only if it still writes the same key at the same width.",
            f"Key: {scope.get('output_key')}",
            f"Width: {scope.get('width')}",
            f"Kind: {scope.get('kind')}",
            f"Contract: {scope.get('contract')}",
            "Pages:",
        ]
        for i, page in enumerate(pages, 1):  # Number them so the model can cite
            body = str(page.get("markdown") or "")[:PAGE_CHARS]
            lines.append(f"[{i}] {page.get('title')} | {page.get('url')}")
            lines.append(body)
        lines.append(
            "Return a JSON list, best first, at most 3 items. "
            "Each item has method, url, sketch. "
            "url must be one of the page URLs above. "
            "If none fit the contract, return []."
        )
        return "\n".join(lines)

    def _ask(self, prompt: str) -> str:
        """One non-streaming chat call. Empty string on failure."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,   # One JSON object, not a stream
                    "format": "json",  # Ask for JSON. Still validated below.
                    "options": {"temperature": 0},  # Same answer on a rerun
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as exc:       # Ollama down is a logged skip
            logging.warning(f"Ollama call failed: {exc}")
            return ""
        body = response.json() or {}
        return str((body.get("message") or {}).get("content") or "").strip()

    @staticmethod
    def _parse(raw: str) -> List[Dict[str, Any]]:
        """A JSON list of dicts, or empty. Never raises."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):     # Some models wrap the list
            data = data.get("methods") or data.get("results") or []
        if not isinstance(data, list):
            return []
        return [row for row in data if isinstance(row, dict)]

    @staticmethod
    def _fits(row: Dict[str, Any], scope: Dict[str, Any]) -> bool:
        """True only if the sketch still names the key and the width."""
        key = str(scope.get("output_key") or "")
        width = str(scope.get("width") or "")
        text = f"{row.get('method', '')} {row.get('sketch', '')}".lower()
        if key and key.lower() not in text:  # Must still emit that key
            return False
        if width and width not in text:  # Must still be that width
            return False
        return bool(row.get("method"))  # A blank method is not a method
