"""Read-only search tools: filtering, truncation notes, errors, and that nothing writes."""

import asyncio
import inspect

import httpx

import ngs360_client
import tools


def attrs(**kw):
    return [{"key": k, "value": v} for k, v in kw.items()]


PROJECTS = [
    {"project_id": "P-3", "name": "Demo: Cardiac Atlas", "attributes": attrs(Assay="scRNA-seq", Status="Active")},
    {"project_id": "P-2", "name": "Demo: Immune Profiling", "attributes": attrs(Assay="RNA-seq", Status="Analysis")},
    {"project_id": "P-1", "name": "Demo: Tumor Exome", "attributes": attrs(Assay="WES", Status="Active")},
]
RUNS = [
    {"run_id": "R-A", "experiment_name": "Demo NovaSeq run 1", "machine_id": "A01234", "flowcell_id": "F1",
     "run_date": "2026-09-01", "status": "Ready"},
    {"run_id": "R-B", "experiment_name": "Demo NextSeq run 2", "machine_id": "VH00321", "flowcell_id": "F2",
     "run_date": "2026-09-10", "status": "In Progress"},
]


class FakeClient:
    def __init__(self, total_projects=None):
        self.total_projects = total_projects

    async def list_projects(self):
        return PROJECTS, self.total_projects or len(PROJECTS)

    async def list_runs(self):
        return RUNS, len(RUNS)

    async def get_run(self, run_id):
        return {**RUNS[0], "run_id": run_id, "run_number": "80", "run_time": "0930",
                "run_folder_uri": "s3://b/x/"}

    async def get_run_sample_count(self, run_id):
        return 16

    async def list_project_samples(self, project_id, skip=0, limit=100):
        rows = [
            {"sample_id": "S1", "attributes": attrs(Tissue="Colon", Condition="Tumor", Index="ACGT-ACGT")},
            {"sample_id": "S2", "attributes": attrs(Tissue="Lung", Condition="Normal", Index="TTTT-AAAA")},
        ]
        return {"data": rows, "total_items": 2}

    async def get_project(self, project_id):
        return {}


def tool(name, client=None):
    return {t.name: t for t in tools.build_tools(client or FakeClient())}[name]


def run(t, **args):
    return asyncio.run(t.ainvoke(args))


def test_search_projects_matches_attribute_values_the_api_does_not_index():
    out = run(tool("search_projects"), query="WES")
    assert "P-1" in out and "P-2" not in out and "Assay=WES" in out


def test_search_projects_requires_every_word_and_ignores_case():
    assert "P-2" in run(tool("search_projects"), query="rna-seq analysis")
    # matching is by substring, so "rna-seq" also finds single-cell "scRNA-seq"
    assert "P-3" in run(tool("search_projects"), query="rna-seq active")
    assert "No projects match" in run(tool("search_projects"), query="wes analysis")


def test_search_projects_empty_query_lists_everything():
    out = run(tool("search_projects"))
    assert out.startswith("3 project(s)") and "P-1" in out and "P-3" in out


def test_search_projects_says_when_the_listing_was_cut():
    out = run(tool("search_projects", FakeClient(total_projects=900)), query="Demo")
    assert "Only the 3 most recent of 900" in out


def test_search_runs_by_status_and_instrument():
    assert "R-B" in run(tool("search_runs"), query="in progress")
    assert "R-A" not in run(tool("search_runs"), query="in progress")
    assert "R-A" in run(tool("search_runs"), query="A01234")
    assert "2026-09-10" in run(tool("search_runs"), query="2026-09-10")


def test_get_run_includes_the_sample_count():
    out = run(tool("get_run"), run_id="R-A")
    assert "samples on this run: 16" in out and "flowcell F1" in out


def test_list_project_samples_filters_and_hides_index_sequences():
    out = run(tool("list_project_samples"), project_id="P-1", filter="tumor colon")
    assert "S1" in out and "S2" not in out and "ACGT" not in out
    assert "1 of 2 sample(s)" in out


def test_tool_errors_are_returned_as_text_not_raised():
    class Broken(FakeClient):
        async def list_runs(self):
            raise httpx.ConnectError("down")

    assert "Could not search runs" in run(tool("search_runs", Broken()))


def test_the_client_only_reads():
    src = inspect.getsource(ngs360_client)
    assert ".post(" not in src and ".put(" not in src and ".delete(" not in src and ".patch(" not in src


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=self)

    def json(self):
        return self._body


def _client_with(monkeypatch, outcomes):
    """NGS360Client whose HTTP layer replays ``outcomes`` (a response or an exception each)."""
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **kw):
            calls.append(url)
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

    monkeypatch.setattr(ngs360_client.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(ngs360_client, "RETRY_DELAY_S", 0)
    return ngs360_client.NGS360Client("tok"), calls


def test_client_retries_a_5xx_once_then_succeeds(monkeypatch):
    c, calls = _client_with(monkeypatch, [_Resp(500), _Resp(200, {"ok": 1})])
    assert asyncio.run(c._get("/runs/search")) == {"ok": 1} and len(calls) == 2


def test_client_retries_a_timeout_once_then_gives_up(monkeypatch):
    c, calls = _client_with(monkeypatch, [httpx.ReadTimeout("t"), httpx.ReadTimeout("t")])
    try:
        asyncio.run(c._get("/runs/search"))
        raise AssertionError("expected a timeout")
    except httpx.ReadTimeout:
        assert len(calls) == 2


def test_client_does_not_retry_a_4xx(monkeypatch):
    c, calls = _client_with(monkeypatch, [_Resp(403)])
    try:
        asyncio.run(c._get("/projects/x"))
        raise AssertionError("expected 403")
    except httpx.HTTPStatusError:
        assert len(calls) == 1
