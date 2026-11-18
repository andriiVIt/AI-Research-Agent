"""OpenAlex works API: filter building, search and metadata parsing. No LLM here."""
from datetime import date

import httpx2

from research.schemas import Comparison, Constraints, Paper

BASE_URL = "https://api.openalex.org"
USER_AGENT = "research-scout/0.1"
WORK_TYPES = "type:article|preprint|review"
SELECT = ("id,doi,display_name,publication_year,cited_by_count,primary_location,"
          "authorships,abstract_inverted_index")
ABSTRACT_LIMIT = 1500


class SourceError(RuntimeError):
    """A bibliographic source failed. The message never contains URLs or keys."""


def range_filter(field: str, comparison: Comparison) -> str:
    """OpenAlex supports only >, < and equality, so inclusive bounds shift by one."""
    op, value = comparison.op, comparison.value
    if op == ">":
        return f"{field}:>{value}"
    if op == ">=":
        return f"{field}:>{value - 1}"
    if op == "<":
        return f"{field}:<{value}"
    if op == "<=":
        return f"{field}:<{value + 1}"
    return f"{field}:{value}"


def build_filter(constraints: Constraints, today: date) -> str:
    parts = [WORK_TYPES]
    year = constraints.effective_year(today)
    if year is not None:
        parts.append(range_filter("publication_year", year))
    if constraints.citations is not None:
        parts.append(range_filter("cited_by_count", constraints.citations))
    return ",".join(parts)


def reconstruct_abstract(inverted_index) -> str:
    if not inverted_index:
        return ""
    positions = [(position, word) for word, places in inverted_index.items() for position in places]
    return " ".join(word for _, word in sorted(positions))


def short_id(openalex_url: str) -> str:
    return openalex_url.rstrip("/").rsplit("/", 1)[-1]


def parse_work(work: dict, retrieved_at: str) -> Paper:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    doi = work.get("doi")
    if doi:
        doi = doi.removeprefix("https://doi.org/")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if len(abstract) > ABSTRACT_LIMIT:
        abstract = abstract[:ABSTRACT_LIMIT].rsplit(" ", 1)[0] + " ..."
    return Paper(openalex_id=short_id(work["id"]), title=work.get("display_name") or "(untitled)",
                 year=work.get("publication_year"), cited_by_count=work.get("cited_by_count") or 0,
                 doi=doi, landing_url=location.get("landing_page_url"),
                 venue=source.get("display_name"),
                 authors=[(item.get("author") or {}).get("display_name", "")
                          for item in (work.get("authorships") or [])[:5]],
                 abstract=abstract, retrieved_at=retrieved_at)


class OpenAlexClient:
    def __init__(self, http: httpx2.AsyncClient, api_key: str = "", today: date | None = None):
        self.http = http
        self._api_key = api_key
        self.today = today or date.today()

    async def _get(self, path: str, params: dict) -> dict:
        if self._api_key:
            params = {**params, "api_key": self._api_key}
        try:
            response = await self.http.get(BASE_URL + path, params=params,
                                           headers={"User-Agent": USER_AGENT})
        except httpx2.HTTPError as error:
            raise SourceError(f"OpenAlex request failed ({type(error).__name__}).") from None
        if response.status_code == 404:
            raise SourceError("OpenAlex has no work with that id.")
        if response.status_code != 200:
            raise SourceError(f"OpenAlex returned HTTP {response.status_code}.")
        return response.json()

    async def search(self, constraints: Constraints, per_page: int = 8):
        """Return (papers, filter string). The filter is safe to log; the key is not part of it."""
        work_filter = build_filter(constraints, self.today)
        payload = await self._get("/works", {"search": constraints.topic, "filter": work_filter,
                                             "per_page": per_page, "select": SELECT})
        retrieved = self.today.isoformat()
        return [parse_work(work, retrieved) for work in payload.get("results", [])], work_filter

    async def get(self, openalex_id: str) -> Paper:
        payload = await self._get(f"/works/{openalex_id}", {"select": SELECT})
        return parse_work(payload, self.today.isoformat())


def evidence_url(openalex_id: str) -> str:
    return f"{BASE_URL}/works/{openalex_id}"
