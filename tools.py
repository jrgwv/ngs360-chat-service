"""
Tools the chat agent can call.

Built per-request via ``build_tools(client)`` so each tool closes over an
``NGS360Client`` carrying the current user's token — that threads the caller's
identity into every REST lookup without global state.

The ``navigate`` tool's side effect (redirecting the browser) is performed by
the frontend, which consumes a ``data-navigate`` SSE part emitted by the
streamer in ``services.py``; the returned string is only what the model sees.
"""

from typing import Any

from langchain_core.tools import tool

from ngs360_client import NGS360Client

NAVIGABLE_TYPES = ("project", "run", "job")

MAX_ROWS = 25  # rows shown to the model per call; the tool says when more matched


def _attrs(item: dict) -> dict[str, str]:
    return {a.get("key"): a.get("value") for a in (item.get("attributes") or []) if a.get("key")}


def _words(text: str) -> list[str]:
    return [w for w in (text or "").lower().replace(",", " ").split() if w]


def _matches(query: str, *haystacks: str) -> bool:
    """Every word of ``query`` appears (case-insensitively) somewhere in the haystacks."""
    text = " ".join(h for h in haystacks if h).lower()
    return all(w in text for w in _words(query))


def _more(shown: int, matched: int, fetched: int, total: int) -> str:
    note = f"\nShowing {shown} of {matched} matches." if matched > shown else ""
    if fetched < total:
        note += f"\nOnly the {fetched} most recent of {total} records were searched."
    return note


# Analysis (workflow) run states reported by WES, and the plain words people use for them.
FAILED_STATES = ("EXECUTOR_ERROR", "SYSTEM_ERROR")
STATE_ALIASES = {
    "failed": FAILED_STATES,
    "failure": FAILED_STATES,
    "failures": FAILED_STATES,
    "error": FAILED_STATES,
    "running": ("RUNNING", "INITIALIZING", "QUEUED"),
    "in progress": ("RUNNING", "INITIALIZING", "QUEUED"),
    "queued": ("QUEUED",),
    "completed": ("COMPLETE",),
    "complete": ("COMPLETE",),
    "succeeded": ("COMPLETE",),
    "successful": ("COMPLETE",),
    "canceled": ("CANCELED", "CANCELING"),
    "cancelled": ("CANCELED", "CANCELING"),
}


def _wes_states(state: str) -> tuple[str, ...] | None:
    """The WES states that a user's word for a state stands for, or None for 'any state'."""
    word = (state or "").strip().lower()
    if not word:
        return None
    return STATE_ALIASES.get(word, (word.upper().replace(" ", "_"),))


def _wes_project(run: dict) -> str:
    return (run.get("tags") or {}).get("ProjectId") or "no project"


def _wes_line(run: dict) -> str:
    name = run.get("name") or (run.get("tags") or {}).get("TaskName") or "unnamed"
    return (
        f"- {run.get('run_id')}: {name}, {run.get('state')}, project {_wes_project(run)}, "
        f"{run.get('start_time') or '?'} to {run.get('end_time') or '?'}"
    )


def build_tools(client: NGS360Client) -> list[Any]:
    @tool
    def navigate(destination: str, id: str) -> str:
        """Take the user to an entity's detail page in NGS360.

        Call this when the user asks to open, go to, or view a specific
        project, run, or job.

        Args:
            destination: One of "project", "run", or "job".
            id: The entity id, e.g. "P-20260507-0008".
        """
        if destination not in NAVIGABLE_TYPES:
            return (
                f"Cannot navigate to '{destination}'. "
                f"Valid destinations are: {', '.join(NAVIGABLE_TYPES)}."
            )
        return f"Navigating to {destination} {id}."

    @tool
    async def lookup_project(project_id: str) -> str:
        """Look up summary information about a project by its id.

        Call this when the user asks about a specific project's status,
        samples, or runs and you need real data to answer.

        Args:
            project_id: The project id, e.g. "P-20260507-0008".
        """
        try:
            data = await client.get_project(project_id)
        except Exception as exc:  # noqa: BLE001 - surface any lookup failure to the model
            return f"Could not fetch project {project_id}: {exc}"
        return f"Project {project_id}: {data}"

    @tool
    async def search_projects(query: str = "") -> str:
        """Search NGS360 projects by words in their name, id, or attributes.

        Use this to list or find projects, for example "RNA-seq", "WES", "Active",
        "Homo sapiens", or a sponsor. Every word must match. An empty query lists the
        newest projects. Returns project id, name and key attributes.

        Args:
            query: Words to look for, e.g. "exome" or "Sequencing WGS". Optional.
        """
        try:
            rows, total = await client.list_projects()
        except Exception as exc:  # noqa: BLE001
            return f"Could not search projects: {exc}"
        hits = [
            p for p in rows
            if _matches(query, p.get("project_id", ""), p.get("name", ""),
                        " ".join(str(v) for v in _attrs(p).values()))
        ]
        if not hits:
            return f"No projects match '{query}' (searched {len(rows)} of {total})."
        lines = []
        for p in hits[:MAX_ROWS]:
            a = _attrs(p)
            keys = ", ".join(f"{k}={v}" for k, v in list(a.items())[:5])
            lines.append(f"- {p['project_id']}: {p.get('name')} ({keys})")
        return f"{len(hits)} project(s):\n" + "\n".join(lines) + _more(min(len(hits), MAX_ROWS), len(hits), len(rows), total)

    @tool
    async def search_runs(query: str = "") -> str:
        """Search SEQUENCING runs (instrument runs on a sequencer, with a flowcell) by words in the run id,
        experiment, instrument, flowcell, date or status. For analysis / workflow runs, such as pipeline
        executions that succeeded or failed, use list_workflow_runs instead.

        Use this for questions like "which runs are in progress", "runs on instrument
        A01234" or "runs from September". Status values are "Ready", "In Progress",
        "Uploading" and "Resync". Every word must match. An empty query lists the newest
        runs.

        Args:
            query: Words to look for, e.g. "in progress", "NovaSeq", "2026-09". Optional.
        """
        try:
            rows, total = await client.list_runs()
        except Exception as exc:  # noqa: BLE001
            return f"Could not search runs: {exc}"
        hits = [
            r for r in rows
            if _matches(query, r.get("run_id", ""), r.get("experiment_name", ""),
                        r.get("machine_id", ""), r.get("flowcell_id", ""),
                        str(r.get("run_date", "")), str(r.get("status", "")))
        ]
        if not hits:
            return f"No runs match '{query}' (searched {len(rows)} of {total})."
        lines = [
            f"- {r['run_id']}: {r.get('experiment_name')}, {r.get('run_date')}, "
            f"instrument {r.get('machine_id')}, status {r.get('status')}"
            for r in hits[:MAX_ROWS]
        ]
        return f"{len(hits)} run(s):\n" + "\n".join(lines) + _more(min(len(hits), MAX_ROWS), len(hits), len(rows), total)

    @tool
    async def get_run(run_id: str) -> str:
        """Get the details of one SEQUENCING run (instrument run), including how many samples are on it.
        For an analysis / workflow run use get_workflow_run.

        Args:
            run_id: The full run id, e.g. "260901_A01234_0080_ALJ91DDSXX".
        """
        try:
            run = await client.get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            return f"Could not fetch run {run_id}: {exc}"
        try:
            samples = str(await client.get_run_sample_count(run_id))
        except Exception:  # noqa: BLE001
            samples = "unknown"
        return (
            f"Run {run['run_id']}: experiment {run.get('experiment_name')}, date {run.get('run_date')} "
            f"{run.get('run_time') or ''}, instrument {run.get('machine_id')}, run number {run.get('run_number')}, "
            f"flowcell {run.get('flowcell_id')}, status {run.get('status')}, folder {run.get('run_folder_uri')}, "
            f"samples on this run: {samples}."
        )

    @tool
    async def list_project_samples(project_id: str, filter: str = "") -> str:
        """List the samples of a project with their attributes (tissue, condition, library kit...).

        Use this for sample questions about a specific project, such as "how many tumor
        samples" or "which samples are colon". Every word of ``filter`` must match a
        sample id or attribute value.

        Args:
            project_id: The project id, e.g. "P-20261102-0001".
            filter: Optional words to narrow the samples, e.g. "Tumor Colon".
        """
        try:
            data = await client.list_project_samples(project_id)
        except Exception as exc:  # noqa: BLE001
            return f"Could not list samples for {project_id}: {exc}"
        rows, total = data.get("data", []), data.get("total_items", 0)
        hits = [
            s for s in rows
            if _matches(filter, s.get("sample_id", ""), " ".join(str(v) for v in _attrs(s).values()))
        ]
        if not hits:
            return f"No samples in {project_id} match '{filter}' ({total} samples in the project)."
        lines = []
        for s in hits[:MAX_ROWS]:
            keys = ", ".join(f"{k}={v}" for k, v in _attrs(s).items() if k not in ("Index",))
            lines.append(f"- {s['sample_id']}: {keys}")
        extra = f"\nOnly the first {len(rows)} of {total} samples were searched." if len(rows) < total else ""
        shown = min(len(hits), MAX_ROWS)
        more = f"\nShowing {shown} of {len(hits)} matches." if len(hits) > shown else ""
        return f"{len(hits)} of {total} sample(s) in {project_id}:\n" + "\n".join(lines) + more + extra

    @tool
    async def get_qc(project_id: str = "", run_id: str = "", failing_only: bool = False) -> str:
        """Get quality-control (QC) metrics for a project's samples or for a sequencing run.

        Use this for questions like "which samples failed QC", "what is the coverage of the
        samples in this project" or "how did the demux of this run look". Give exactly one of
        ``project_id`` (per-sample metrics) or ``run_id`` (run-level demux metrics). Samples
        with ``qc_status`` FAIL are listed first.

        Args:
            project_id: The project id, e.g. "P-20261102-0001".
            run_id: The full run id, e.g. "260901_A01234_0080_ALJ91DDSXX".
            failing_only: If true, list only samples whose qc_status is FAIL.
        """
        if bool(project_id) == bool(run_id):
            return "Give exactly one of project_id or run_id."
        scope = project_id or run_id
        try:
            data = await client.search_qc(project_id=project_id, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            return f"Could not fetch QC for {scope}: {exc}"
        records = data.get("data", [])
        if not records:
            return f"No QC records for {scope}."
        rec = records[0]
        meta = ", ".join(f"{m['key']}={m['value']}" for m in rec.get("metadata", []) if m.get("key") != "synthetic")
        rows = []
        for m in rec.get("metrics", []):
            vals = {v["key"]: v["value"] for v in m.get("values", [])}
            names = ", ".join(x["sample_name"] for x in m.get("samples", [])) or m.get("name", "")
            rows.append((names, vals))
        failed = [r for r in rows if str(r[1].get("qc_status", "")).upper() == "FAIL"]
        header = f"QC for {scope} (record {rec.get('created_on', '')[:10]}, {meta}): {len(rows)} metric group(s)"
        if any("qc_status" in v for _, v in rows):
            header += f", {len(failed)} failing"
        shown = failed if failing_only else failed + [r for r in rows if r not in failed]
        if failing_only and not failed:
            return header + "\nNo failing samples."
        lines = [
            f"- {n}: " + ", ".join(f"{k}={v}" for k, v in list(vals.items())[:9])
            for n, vals in shown[:MAX_ROWS]
        ]
        more = f"\nShowing {min(len(shown), MAX_ROWS)} of {len(shown)}." if len(shown) > MAX_ROWS else ""
        return header + "\n" + "\n".join(lines) + more

    @tool
    async def list_workflow_runs(state: str = "", project_id: str = "") -> str:
        """List ANALYSIS (workflow) runs, meaning pipeline executions on the compute platform, newest first.

        Use this for "which analyses failed", "what is running", "runs for project P-..." and similar. These are
        different from sequencing runs (instrument runs), which search_runs covers.

        Args:
            state: Optional. "failed", "running", "completed", "canceled", or an exact state such as
                EXECUTOR_ERROR, COMPLETE, RUNNING, QUEUED.
            project_id: Optional project id, e.g. "P-20261102-0004", to show only that project's runs.
        """
        try:
            runs = await client.list_wes_runs()
        except Exception as exc:  # noqa: BLE001
            return f"Could not list analysis runs: {exc}"
        states = _wes_states(state)
        hits = [
            r for r in runs
            if (states is None or str(r.get("state", "")).upper() in states)
            and (not project_id or _wes_project(r).lower() == project_id.strip().lower())
        ]
        hits.sort(key=lambda r: str(r.get("start_time") or ""), reverse=True)
        if not hits:
            return f"No analysis runs match state '{state}' and project '{project_id}' (searched {len(runs)})."
        lines = [_wes_line(r) for r in hits[:MAX_ROWS]]
        more = f"\nShowing {MAX_ROWS} of {len(hits)} matches." if len(hits) > MAX_ROWS else ""
        return f"{len(hits)} analysis run(s):\n" + "\n".join(lines) + more

    @tool
    async def get_workflow_run(run_id: str) -> str:
        """Get the details of one ANALYSIS (workflow) run: state, project, name, times, exit code, where its
        outputs go. To learn why a run failed use explain_workflow_failure.

        Args:
            run_id: The analysis run id from list_workflow_runs, e.g. "8b2e402a-951a-4338-8a07-7ca339ab060b".
        """
        try:
            run = await client.get_wes_run(run_id)
        except Exception as exc:  # noqa: BLE001
            return f"Could not fetch analysis run {run_id}: {exc}"
        req = run.get("request") or {}
        log = run.get("run_log") or {}
        params = req.get("workflow_engine_parameters") or {}
        return (
            f"Analysis run {run.get('run_id')}: name {run.get('name')}, state {run.get('state')}, "
            f"project {_wes_project({'tags': req.get('tags')})}, started {log.get('start_time')}, "
            f"ended {log.get('end_time')}, exit code {log.get('exit_code')}, "
            f"output location {params.get('outputUri')}, "
            f"workflow {req.get('workflow_url')}."
        )

    @tool
    async def explain_workflow_failure(run_id: str) -> str:
        """Explain WHY an analysis (workflow) run failed, using the platform's own failure reason, the failed
        task and the tail of the engine log. Use this for "why did run X fail" instead of guessing.

        Args:
            run_id: The analysis run id from list_workflow_runs.
        """
        try:
            # Fetched with the caller's own token, so only runs this user may see can be explained.
            run = await client.get_wes_run(run_id)
        except Exception as exc:  # noqa: BLE001
            return f"Could not fetch analysis run {run_id}: {exc}"
        state = str(run.get("state"))
        if state not in FAILED_STATES and state != "CANCELED":
            return f"Run {run_id} is {state}, not failed, so there is no failure to explain."
        project = _wes_project({"tags": (run.get("request") or {}).get("tags")})
        header = f"Run {run_id} ({run.get('name')}, project {project}) ended in {state}"
        try:
            # The WES service reads the execution engine itself, with the same authorization as the call above.
            info = await client.get_wes_diagnosis(run_id)
        except Exception as exc:  # noqa: BLE001 - report the class only, never raw error text
            return f"{header}. Its failure details could not be read ({type(exc).__name__})."
        if not info.get("available"):
            return f"{header}. {info.get('note') or 'No further detail is available.'}"
        parts = [
            f"{header}. HealthOmics run {info.get('engine_run_id')} is {info.get('engine_status')}.",
            f"Failure reason: {info.get('failure_reason') or 'not reported'}.",
            f"Message: {info.get('message') or 'none'}",
        ]
        for task in info.get("failed_tasks") or []:
            parts.append(f"Failed task {task.get('name')}: {task.get('status')} {task.get('message') or ''}".strip())
        if info.get("log_tail"):
            parts.append("Last engine log lines:\n" + "\n".join(f"  {line}" for line in info["log_tail"][-12:]))
        return "\n".join(parts)

    return [navigate, lookup_project, search_projects, search_runs, get_run, list_project_samples, get_qc,
            list_workflow_runs, get_workflow_run, explain_workflow_failure]
