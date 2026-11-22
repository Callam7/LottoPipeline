# Modified By: Callam
# Project: Lotto Generator
# Purpose: Stage 2 retrieve — paper and docs filter
# Description:
#   - Takes queries and similar_urls from the brief. Never opens a project file.
#   - Lane order: OpenAlex, then arXiv, then Exa.
#   - Drops a paper with no abstract. Drops a hit with no URL.
#   - Merges on a normalized URL. Caps the list for extract.py.
#   - Does not call Jina. Does not call Ollama.

import logging                         # Same stamp format as the rest of the adaptor
import os                              # Reads OPENALEX_API_KEY / EXA_API_KEY from the environment
from typing import Any, Dict, List, Optional  # Hit dicts and the scope slice
from urllib.parse import urlsplit, urlunsplit  # Normalize before the merge

import arxiv                           # Preprint lane. No key.
import requests                        # OpenAlex HTTP. No OpenAlex package.
from dotenv import load_dotenv         # .env in the project root
from exa_py import Exa                 # Docs lane. Key from EXA_API_KEY.

load_dotenv()                          # So a menu run sees the same keys as a one-liner

logging.basicConfig(                   # One log format for this module
    level=logging.INFO,                # INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Time + level + text
)

OPENALEX_URL = "https://api.openalex.org/works"  # Paper search. Key is a query param.
OPENALEX_PER_QUERY = 5                 # Cap per query before the merge
ARXIV_MAX_RESULTS = 5                  # One arXiv call, not one per query
EXA_PER_QUERY = 3                      # Docs hits per query
EXA_SIMILAR = 3                        # find_similar cap, if a docs URL exists
MERGED_CAP = 9                         # What extract.py is allowed to read
REQUEST_TIMEOUT = 30                   # Seconds. A hung lane must not hang the menu.


class Retrieve:
    """
    Filter the brief into a capped hit list.

    Input is the scope dict. Only queries and similar_urls are read.
    Output is a list of hit dicts for extract.py. Empty list means no_viable.
    """

    def __init__(self) -> None:
        self.openalex_key = os.getenv("OPENALEX_API_KEY") or ""  # Optional. Free key raises the budget.
        self.exa = Exa()               # Reads EXA_API_KEY. Raises if the key is missing.
        self.arxiv_client = arxiv.Client(  # One client. delay_seconds keeps us polite.
            page_size=ARXIV_MAX_RESULTS,
            delay_seconds=3.0,
            num_retries=2,
        )

    def run(self, scope: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAlex, then arXiv, then Exa. Merge. Cap. No Jina. No Ollama."""
        queries = self._queries(scope)  # 2–4 phrases from the brief
        if not queries:                # Nothing to search
            logging.info("Retrieve skipped: no queries.")
            return []
        similar = self._similar_urls(scope)  # Docs URLs. May be empty.

        papers = self._openalex(queries)  # Lane 1. Abstract required.
        papers.extend(self._arxiv(queries))  # Lane 2. Summary required.
        docs = self._exa(queries, similar)  # Lane 3. URL required.

        merged = self._merge(papers + docs)  # One row per normalized URL
        openalex_hits = [h for h in merged if "openalex" in h.get("sources", [])][:4]
        arxiv_hits = [                 # arXiv only. A dual-source row is already in the four.
            h for h in merged
            if "arxiv" in h.get("sources", []) and h not in openalex_hits
        ][:3]
        exa_hits = [                   # Docs only. A page already kept is not counted again.
            h for h in merged
            if "exa" in h.get("sources", [])
            and h not in openalex_hits
            and h not in arxiv_hits
        ][:2]
        capped = (openalex_hits + arxiv_hits + exa_hits)[:MERGED_CAP]  # 4 + 3 + 2.
        logging.info(f"Retrieve ready: {len(capped)} hit(s) from {len(queries)} queries.")
        return capped                  # [] is no_viable for the next stage

    def _queries(self, scope: Optional[Dict[str, Any]]) -> List[str]:
        """The only search text this class accepts."""
        if not scope:                  # Brief returned None
            return []
        raw = scope.get("queries") or []  # Brief key
        return [q for q in raw if isinstance(q, str) and q.strip()]  # Drop blanks

    def _similar_urls(self, scope: Optional[Dict[str, Any]]) -> List[str]:
        """Docs pages from the brief. find_similar uses the first one."""
        if not scope:
            return []
        raw = scope.get("similar_urls") or []  # May be empty
        return [u for u in raw if isinstance(u, str) and u.startswith("http")]

    def _openalex(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Lane 1. Method phrases only. Abstract and URL required."""
        hits: List[Dict[str, Any]] = []  # Papers that survived this lane
        for query in queries:          # One search per brief phrase
            params = {                 # Relevant work with an abstract
                "search": query,       # variational quantum feature map, etc.
                "filter": "has_abstract:true",  # Drop abstract-less works up front
                "sort": "relevance_score:desc",  # Method match first
                "per_page": OPENALEX_PER_QUERY,
            }
            if self.openalex_key:      # Free key raises the daily budget
                params["api_key"] = self.openalex_key
            try:
                response = requests.get(  # No OpenAlex package
                    OPENALEX_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()  # 4xx / 5xx become a logged skip
                payload = response.json()  # Dict, or not
                works = payload.get("results") if isinstance(payload, dict) else None
                works = works or []    # Page of works
            except Exception as exc:   # One bad query does not kill the lane
                logging.warning(f"OpenAlex failed for '{query}': {exc}")
                continue
            for work in works:         # Keep only a usable paper
                if not isinstance(work, dict):  # Skip a malformed row
                    continue
                abstract = self._abstract_from_index(  # Inverted index -> text
                    work.get("abstract_inverted_index")
                )
                if not abstract.strip():  # Rule: no abstract, no hit
                    continue
                if not self._matches_query(str(work.get("display_name") or ""), abstract, query):
                    continue           # Not about this method
                raw_url = work.get("doi") or work.get("id") or ""  # DOI, else OpenAlex id
                url = self._normalize_url(str(raw_url))  # Clean before the merge
                if not url.startswith("https://"):  # Rule: no URL, no hit
                    continue
                year = work.get("publication_year")  # int, or missing
                citations = work.get("cited_by_count")  # int, or missing
                hits.append({          # One paper row for extract.py
                    "url": url,
                    "title": str(work.get("display_name") or ""),
                    "source": "openalex",
                    "year": year if isinstance(year, int) else None,
                    "citations": citations if isinstance(citations, int) else None,
                    "abstract": abstract,
                    "snippet": None,   # Exa fills this. Papers do not.
                    "query": query,    # Which method phrase found it
                })
        logging.info(f"OpenAlex: {len(hits)} hit(s).")
        return hits

    def _arxiv(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Lane 2. One call for every brief phrase. Summary and URL required."""
        quoted = [f'all:"{q.replace(chr(34), "")}"' for q in queries]  # Phrase, no embedded quotes
        search = arxiv.Search(         # One request. Do not download the PDF.
            query=" OR ".join(quoted),  # Same method phrases as OpenAlex
            max_results=ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        hits: List[Dict[str, Any]] = []  # Preprints that survived this lane
        try:
            results = list(self.arxiv_client.results(search))  # Generator -> list
        except Exception as exc:       # Empty page or HTTP error. Lane is optional.
            logging.warning(f"arXiv failed: {exc}")
            return []
        for paper in results:          # arxiv.Result
            abstract = (paper.summary or "").strip()  # Abstract is summary
            if not abstract:           # Rule: no abstract, no hit
                continue
            blob_title = (paper.title or "").strip()
            if not any(self._matches_query(blob_title, abstract, q) for q in queries):
                continue           # Not about any brief phrase
            url = self._normalize_url(paper.entry_id or "")  # abs URL, not the PDF
            if not url.startswith("https://"):  # Rule: no URL, no hit
                continue
            year = paper.published.year if paper.published else None  # datetime -> year
            hits.append({              # One preprint row for extract.py
                "url": url,
                "title": (paper.title or "").strip(),
                "source": "arxiv",
                "year": year,
                "citations": None,     # arXiv does not return a citation count
                "abstract": abstract,
                "snippet": None,       # Exa fills this. Papers do not.
                "query": " OR ".join(queries),  # One call covered every phrase
            })
        logging.info(f"arXiv: {len(hits)} hit(s).")
        return hits

    def _exa(self, queries: List[str], similar_urls: List[str]) -> List[Dict[str, Any]]:
        """Lane 3. Docs. URL required. Term filter same as the paper lanes."""
        hits: List[Dict[str, Any]] = []  # Docs that survived this lane
        for query in queries:          # One search per brief phrase
            try:
                result = self.exa.search(  # Highlights only. No full page text.
                    query,
                    num_results=EXA_PER_QUERY,
                    contents={"highlights": True},
                )
            except Exception as exc:   # One bad query does not kill the lane
                logging.warning(f"Exa search failed for '{query}': {exc}")
                continue
            hits.extend(self._exa_rows(result, query))  # Filter, then keep
        if similar_urls:               # find_similar on the first docs URL only
            try:
                result = self.exa.find_similar(
                    similar_urls[0],
                    num_results=EXA_SIMILAR,
                    contents={"highlights": True},
                )
                hits.extend(self._exa_rows(result, similar_urls[0]))
            except Exception as exc:   # Docs URL is optional. Papers still stand.
                logging.warning(f"Exa find_similar failed: {exc}")
        logging.info(f"Exa: {len(hits)} hit(s).")
        return hits

    def _exa_rows(self, result: Any, query: str) -> List[Dict[str, Any]]:
        """Keep a docs row only if it has a URL and matches the method."""
        rows: List[Dict[str, Any]] = []  # Rows from one Exa response
        for item in getattr(result, "results", []) or []:  # SDK result list
            url = self._normalize_url(getattr(item, "url", "") or "")  # Clean before the merge
            if not url.startswith("https://"):  # Rule: no URL, no hit
                continue
            title = getattr(item, "title", "") or ""  # Page title
            highlights = getattr(item, "highlights", None) or []  # Query-relevant excerpts
            snippet = " ".join(str(h) for h in highlights).strip()  # One string for the filter
            if not self._matches_query(title, snippet, query):  # Same bar as the paper lanes
                continue
            rows.append({              # One docs row for extract.py
                "url": url,
                "title": title,
                "source": "exa",
                "year": None,          # Exa highlight rows do not carry a year here
                "citations": None,     # Docs are not citation-ranked
                "abstract": None,      # Docs have a snippet, not an abstract
                "snippet": snippet or None,
                "query": query,        # Phrase, or the similar URL, that found it
            })
        return rows

    def _merge(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One row per normalized URL. A paper row wins over a docs duplicate."""
        merged: Dict[str, Dict[str, Any]] = {}  # normalized url -> hit
        for hit in hits:               # Papers first in run(), so they land first
            url = self._normalize_url(str(hit.get("url") or ""))  # Same key as the lanes
            if not url.startswith("https://"):  # No URL, no row
                continue
            current = merged.get(url)  # Already seen this page?
            if current is None:        # First sighting
                hit = dict(hit)        # Do not mutate the lane list
                hit["url"] = url
                hit["sources"] = [hit.get("source")]  # Both lanes can name one page
                merged[url] = hit
                continue
            source = hit.get("source")  # openalex / arxiv / exa
            if source and source not in current["sources"]:
                current["sources"].append(source)  # Same page, second lane
            if not current.get("abstract") and hit.get("abstract"):
                current["abstract"] = hit["abstract"]  # Fill a missing abstract
            if not current.get("snippet") and hit.get("snippet"):
                current["snippet"] = hit["snippet"]  # Fill a missing snippet
            if current.get("citations") is None and hit.get("citations") is not None:
                current["citations"] = hit["citations"]  # OpenAlex count wins if present
        return list(merged.values())  # Insertion order. Cap happens in run().

    @staticmethod
    def _normalize_url(url: str) -> str:
        """https, no trailing slash, arXiv pdf collapsed to abs, DOI lowercased."""
        if not url:                    # Nothing to clean
            return ""
        text = url.strip()             # Drop surrounding space
        if text.lower().startswith("http://"):  # One scheme
            text = "https://" + text[7:]
        if not text.lower().startswith("https://"):  # Not a URL we can merge on
            return ""
        parts = urlsplit(text)         # scheme, netloc, path, query, fragment
        host = (parts.netloc or "").lower()  # Host compare is case-insensitive
        path = (parts.path or "").rstrip("/")  # Trailing slash is not a different page
        if host.endswith("arxiv.org") and "/pdf/" in path:  # pdf and abs are one paper
            paper_id = path.split("/pdf/", 1)[1]  # 1234.5678v1.pdf
            if paper_id.lower().endswith(".pdf"):
                paper_id = paper_id[:-4]  # Drop the extension
            path = "/abs/" + paper_id
        if host.endswith("doi.org"):   # DOI paths are case-insensitive
            path = path.lower()
        if not host or not path:       # scheme:// with nothing after it
            return ""
        return urlunsplit(("https", host, path, "", ""))  # Drop query and fragment

    @staticmethod
    def _matches_query(title: str, abstract: str, query: str) -> bool:
        """True if the hit is about the method, not a famous loose match."""
        stop = {                      # Words that do not identify a method
            "a", "an", "the", "of", "for", "and", "or", "with", "using",
            "based", "from", "to", "in", "on", "by", "alternatives",
            "features", "normalisation", "normalization",
        }
        words = [w for w in query.lower().split() if w not in stop and len(w) > 2]
        if not words:                  # Nothing left to require
            return False
        text = f"{title} {abstract}".lower()  # Title and abstract, or title and snippet
        found = sum(1 for w in words if w in text)  # How many method words landed
        need = 1 if len(words) == 1 else 2  # One word, or two from a phrase
        return found >= need           # A news page fails. A method page passes.

    @staticmethod
    def _abstract_from_index(index: Optional[Dict[str, List[int]]]) -> str:
        """Rebuild an OpenAlex abstract. Empty string means drop the hit."""
        if not isinstance(index, dict) or not index:  # Missing or wrong type
            return ""
        placed: List[tuple] = []       # (position, word)
        for word, positions in index.items():  # word -> list of indexes
            if not isinstance(positions, list):  # Malformed entry
                continue
            for pos in positions:
                if isinstance(pos, int):  # Position must be an index
                    placed.append((pos, str(word)))
        if not placed:                 # Nothing usable
            return ""
        placed.sort(key=lambda pair: pair[0])  # Reading order
        return " ".join(word for _, word in placed)  # Plain abstract
