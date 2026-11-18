"""Crossref citation cross-check by DOI (no key needed). No LLM here."""
from urllib.parse import quote

import httpx2

from research.services.openalex import SourceError, USER_AGENT

BASE_URL = "https://api.crossref.org"


def work_url(doi: str) -> str:
    return f"{BASE_URL}/works/{quote(doi, safe='/')}"


class CrossrefClient:
    def __init__(self, http: httpx2.AsyncClient):
        self.http = http

    async def citation_count(self, doi: str):
        """Crossref's is-referenced-by-count, or None when Crossref does not know the DOI."""
        try:
            response = await self.http.get(work_url(doi), headers={"User-Agent": USER_AGENT})
        except httpx2.HTTPError as error:
            raise SourceError(f"Crossref request failed ({type(error).__name__}).") from None
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise SourceError(f"Crossref returned HTTP {response.status_code}.")
        return response.json().get("message", {}).get("is-referenced-by-count")
