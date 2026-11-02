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
        """Search sequencing runs by words in the run id, experiment, instrument, flowcell,
        date or status.

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
        """Get the details of one sequencing run, including how many samples are on it.

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

    return [navigate, lookup_project, search_projects, search_runs, get_run, list_project_samples]
