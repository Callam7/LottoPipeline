# Modified By: Callam
# Project: Lotto Generator
# Purpose: Stage 2 extract — read the capped hits
# Description:
#   - Takes the hit list from retrieve.py. Never opens a project file.
#   - Allowlists the host before any fetch. A page off the list is dropped.
#   - Sends the URL to Jina Reader and attaches the markdown.
#   - Drops a page that comes back empty or fails.
#   - Does not call Ollama. Does not search.

import logging                         # Same stamp format as the rest of the adaptor
import os                              # Reads JINA_API_KEY
from typing import Any, Dict, List     # Hit dicts
from urllib.parse import urlsplit     # Host check before the fetch

import requests                        # Jina is a GET. No Jina package.
from dotenv import load_dotenv         # .env in the project root

load_dotenv()                          # Same keys the menu already uses

logging.basicConfig(                   # One log format for this module
    level=logging.INFO,                # INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Time + level + text
)

JINA_PREFIX = "https://r.jina.ai/"    # Reader. Prepend this to the page URL.
REQUEST_TIMEOUT = 60                   # Seconds. Jina is slower than a search call.
MARKDOWN_CAP = 8000                    # Chars. The judge does not need the whole page.
ALLOWED_HOSTS = (                      # Fetch these. Drop everything else.
    "arxiv.org",
    "doi.org",
    "github.com",
    "scikit-learn.org",
    "pennylane.ai",
    "openreview.net",
    "readthedocs.io",
)


class Extract:
    """
    Read the capped hit list.

    Input is the list from Retrieve.run. Output is the same rows with markdown
    attached. A row that cannot be read is dropped, not kept empty.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("JINA_API_KEY") or ""  # Free key raises the rate limit.

    def run(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Allowlist, fetch, attach. No search. No Ollama."""
        if not hits:                   # Nothing to read
            logging.info("Extract skipped: no hits.")
            return []
        if not self.api_key:           # A missing key is a setup error, not an empty page
            logging.warning("Extract skipped: JINA_API_KEY is missing.")
            return []
        kept: List[Dict[str, Any]] = []  # Rows the judge is allowed to see
        for hit in hits:               # One page at a time. Order is the retrieve order.
            url = str(hit.get("url") or "")
            if not self._allowed(url):  # Off the list. Do not fetch.
                logging.info(f"Extract dropped, host not allowed: {url}")
                continue
            markdown = self._read(url)  # Jina call
            if not markdown:           # Empty or failed. Drop the row.
                continue
            row = dict(hit)            # Do not mutate the retrieve list
            row["markdown"] = markdown[:MARKDOWN_CAP]  # Cap before the judge
            kept.append(row)
        logging.info(f"Extract ready: {len(kept)} page(s) from {len(hits)} hit(s).")
        return kept                    # [] is no_viable for the judge

    def _allowed(self, url: str) -> bool:
        """True if the host is on the allowlist. No fetch happens before this."""
        if not url.startswith("https://"):  # Same bar as retrieve
            return False
        host = (urlsplit(url).netloc or "").lower()  # example.com, or www.example.com
        if host.startswith("www."):    # www is not a different site
            host = host[4:]
        return any(host == name or host.endswith("." + name) for name in ALLOWED_HOSTS)

    def _read(self, url: str) -> str:
        """One Jina call. Empty string means drop the hit."""
        try:
            response = requests.get(   # Proven shape. Do not set X-Timeout.
                JINA_PREFIX + url,     # r.jina.ai/https://...
                headers={
                    "Authorization": f"Bearer {self.api_key}",  # Free key, higher limit
                    "X-Respond-With": "markdown",  # Page body, not a screenshot
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()  # 4xx / 5xx become a logged skip
        except Exception as exc:       # One bad page does not kill the lane
            logging.warning(f"Jina failed for '{url}': {exc}")
            return ""
        text = (response.text or "").strip()  # Plain markdown, or a titled block
        return self._body(text)        # Drop the Title / URL Source header if present

    @staticmethod
    def _body(text: str) -> str:
        """The markdown after Jina's header, or the whole text if there is none."""
        marker = "Markdown Content:"  # Header the plain response uses
        if marker in text:
            return text.split(marker, 1)[1].strip()  # Page only
        return text                    # Already just the page