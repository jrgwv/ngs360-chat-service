"""
Thin async HTTP client for the NGS360 REST API.

The agent's tools use this to fetch real data. Every request carries the
caller's bearer token, so the REST API enforces authorization — this service
never touches the database.
"""

import httpx

from config import get_settings


class NGS360Client:
    def __init__(self, token: str) -> None:
        settings = get_settings()
        self._base = settings.NGS360_API_URL.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    async def _get(self, path: str, params: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base}{path}", headers=self._headers, params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_project(self, project_id: str) -> dict:
        # TODO: confirm the exact REST path/shape against the API's project router.
        return await self._get(f"/projects/{project_id}")

    # ---- read-only listings used by the search tools ------------------------------------
    # The API's text search matches names/ids but not attribute values or run status, so the
    # tools fetch a bounded, unfiltered listing and filter it themselves. These methods return
    # (rows, total) where total is the server's count, so callers can say when a listing was cut.

    async def list_projects(self, max_rows: int = 300) -> tuple[list[dict], int]:
        return await self._list("/projects/search", {"query": "", "sort_by": "project_id", "sort_order": "desc"}, max_rows)

    async def list_runs(self, max_rows: int = 300) -> tuple[list[dict], int]:
        return await self._list("/runs/search", {"query": "", "sort_by": "run_date", "sort_order": "desc"}, max_rows)

    async def _list(self, path: str, params: dict, max_rows: int) -> tuple[list[dict], int]:
        rows: list[dict] = []
        total = 0
        page = 1
        while len(rows) < max_rows:
            data = await self._get(path, {**params, "page": page, "per_page": 100})
            total = data.get("total_items", 0)
            rows.extend(data.get("data", []))
            if not data.get("has_next"):
                break
            page += 1
        return rows[:max_rows], total

    async def get_run(self, run_id: str) -> dict:
        return await self._get(f"/runs/{run_id}")

    async def get_run_sample_count(self, run_id: str) -> int:
        return len(await self._get(f"/runs/{run_id}/samples"))

    async def list_project_samples(self, project_id: str, skip: int = 0, limit: int = 100) -> dict:
        return await self._get(f"/projects/{project_id}/samples", {"skip": skip, "limit": limit})
