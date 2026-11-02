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


WES_RUNS = [
    {"run_id": "w-ok", "state": "COMPLETE", "name": "demo-hello", "start_time": "2026-11-02T14:51:58Z",
     "end_time": "2026-11-02T14:58:31Z", "tags": {"ProjectId": "P-4"}},
    {"run_id": "w-bad", "state": "EXECUTOR_ERROR", "name": "sandbox-hello", "start_time": "2026-10-31T01:05:05Z",
     "end_time": "2026-10-31T01:10:29Z", "tags": {"ProjectId": "P-2"}},
    {"run_id": "w-old", "state": "SYSTEM_ERROR", "name": "sandbox-hello", "start_time": "2026-10-30T10:00:00Z",
     "end_time": "2026-10-30T10:03:00Z", "tags": {"ProjectId": "P-1"}},
]
DIAGNOSES = {
    "w-bad": {"run_id": "w-bad", "available": True, "engine_run_id": "2933968", "engine_status": "FAILED",
              "failure_reason": "CANNOT_START_CONTAINER_ERROR", "message": "Unable to start workflow task: main.cwl",
              "failed_tasks": [{"name": "main.cwl", "status": "FAILED", "message": ""}],
              "log_tail": ["engine: container exited"], "note": None,
              "hint": "Add bash to the image or use a bash-based image."},
    "w-nolink": {"run_id": "w-nolink", "available": False, "engine_run_id": None,
                 "note": "This run has no execution-engine run ID yet, so there is nothing more to read."},
}
WES_FULL = {
    "w-ok": {"run_id": "w-ok", "state": "COMPLETE", "name": "demo-hello",
             "request": {"tags": {"ProjectId": "P-4"}, "workflow_url": "wf:1",
                         "workflow_engine_parameters": {"outputUri": "s3://out/P-4/"}},
             "run_log": {"start_time": "t0", "end_time": "t1", "exit_code": 0},
             "outputs": {}},
    "w-bad": {"run_id": "w-bad", "state": "EXECUTOR_ERROR", "name": "sandbox-hello",
              "request": {"tags": {"ProjectId": "P-2"}, "workflow_url": "wf:2", "workflow_engine_parameters": {}},
              "run_log": {"start_time": "t0", "end_time": "t1", "exit_code": 1},
              "outputs": {}},
    "w-nolink": {"run_id": "w-nolink", "state": "EXECUTOR_ERROR", "name": "x", "request": {"tags": {}},
                 "run_log": {}, "outputs": {}},
    "w-denied": {"run_id": "w-denied", "state": "EXECUTOR_ERROR", "name": "y", "request": {"tags": {}},
                 "run_log": {}, "outputs": {}},
}


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

    async def list_wes_runs(self):
        return WES_RUNS

    async def get_wes_run(self, run_id):
        if run_id not in WES_FULL:
            raise httpx.HTTPStatusError("404", request=None, response=None)
        return WES_FULL[run_id]

    async def get_wes_diagnosis(self, run_id):
        self.diagnosed = getattr(self, "diagnosed", []) + [run_id]
        if run_id == "w-denied":
            raise RuntimeError("secret internal detail")
        return DIAGNOSES[run_id]

    async def search_qc(self, project_id="", run_id=""):
        def m(name, status, cov):
            return {"name": "sample_qc", "samples": [{"sample_name": name}],
                    "values": [{"key": "mean_coverage", "value": cov}, {"key": "qc_status", "value": status}]}
        rec = {"created_on": "2026-11-02T10:00:00", "metadata": [{"key": "pipeline", "value": "wgs"}, {"key": "synthetic", "value": "true"}],
               "metrics": [m("S1", "PASS", 40.1), m("S2", "FAIL", 11.2)]}
        return {"data": [rec] if (project_id or run_id) != "EMPTY" else []}


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


def test_get_qc_lists_failures_first():
    out = asyncio.run(tool("get_qc").ainvoke({"project_id": "P-1"}))
    assert "1 failing" in out and out.index("S2") < out.index("S1")
    assert "synthetic" not in out


def test_get_qc_failing_only_and_empty_and_args():
    out = asyncio.run(tool("get_qc").ainvoke({"project_id": "P-1", "failing_only": True}))
    assert "S2" in out and "S1" not in out
    assert "No QC records" in asyncio.run(tool("get_qc").ainvoke({"project_id": "EMPTY"}))
    assert "exactly one" in asyncio.run(tool("get_qc").ainvoke({}))
    assert "exactly one" in asyncio.run(tool("get_qc").ainvoke({"project_id": "a", "run_id": "b"}))


# ---- analysis (workflow) runs and failure explanation --------------------------------------------

def test_list_workflow_runs_failed_means_executor_and_system_errors_newest_first():
    out = asyncio.run(tool("list_workflow_runs").ainvoke({"state": "failed"}))
    assert "2 analysis run(s)" in out and "w-ok" not in out
    assert out.index("w-bad") < out.index("w-old")  # newest first


def test_list_workflow_runs_state_words_project_filter_and_empty():
    assert "w-ok" in asyncio.run(tool("list_workflow_runs").ainvoke({"state": "completed"}))
    assert "w-ok" in asyncio.run(tool("list_workflow_runs").ainvoke({"state": "complete"}))
    out = asyncio.run(tool("list_workflow_runs").ainvoke({"project_id": "p-2"}))
    assert "w-bad" in out and "w-ok" not in out  # case-insensitive project match
    assert "w-ok" in asyncio.run(tool("list_workflow_runs").ainvoke({"state": "COMPLETE", "project_id": "P-4"}))
    assert "No analysis runs match" in asyncio.run(tool("list_workflow_runs").ainvoke({"state": "running"}))
    assert len(asyncio.run(tool("list_workflow_runs").ainvoke({})).splitlines()) == 4  # header + 3 runs


def test_get_workflow_run_summarises_the_run():
    out = asyncio.run(tool("get_workflow_run").ainvoke({"run_id": "w-ok"}))
    assert "state COMPLETE" in out and "project P-4" in out and "exit code 0" in out
    assert "s3://out/P-4/" in out
    assert "Could not fetch" in asyncio.run(tool("get_workflow_run").ainvoke({"run_id": "missing"}))


def test_explain_failure_uses_the_reason_from_the_wes_service():
    client = FakeClient()
    out = asyncio.run(tool("explain_workflow_failure", client).ainvoke({"run_id": "w-bad"}))
    assert client.diagnosed == ["w-bad"]
    assert "HealthOmics run 2933968 is FAILED" in out
    assert "CANNOT_START_CONTAINER_ERROR" in out and "Unable to start workflow task" in out
    assert "Failed task main.cwl" in out and "engine: container exited" in out
    assert "Likely next step: Add bash to the image" in out


def test_explain_failure_skips_healthy_runs_reports_missing_details_and_hides_raw_errors():
    client = FakeClient()
    ok = asyncio.run(tool("explain_workflow_failure", client).ainvoke({"run_id": "w-ok"}))
    assert "not failed" in ok and not getattr(client, "diagnosed", [])  # a healthy run triggers no engine lookup
    nolink = asyncio.run(tool("explain_workflow_failure", client).ainvoke({"run_id": "w-nolink"}))
    assert "no execution-engine run ID" in nolink
    denied = asyncio.run(tool("explain_workflow_failure", client).ainvoke({"run_id": "w-denied"}))
    assert "could not be read (RuntimeError)" in denied and "secret internal detail" not in denied
    unknown = asyncio.run(tool("explain_workflow_failure", client).ainvoke({"run_id": "nope"}))
    assert "Could not fetch" in unknown and client.diagnosed == ["w-nolink", "w-denied"]  # unknown id never diagnosed
