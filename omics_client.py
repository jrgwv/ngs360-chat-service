"""
Read-only access to AWS HealthOmics run details, used to explain why an analysis run failed.

The WES record of a failed run says only that it failed and links to CloudWatch; the reason (for example
CANNOT_START_CONTAINER_ERROR and the message that goes with it) lives in HealthOmics. This module fetches it.

Authorization: the HealthOmics run id is never taken from the user's text. ``omics_run_id_from_wes`` extracts it
from a WES run record that the caller already fetched with the user's own token, so only runs the user may see
through WES can be explained. The service's IAM role is read-only (GetRun, ListRunTasks, GetLogEvents).
"""

import asyncio
import re
from typing import Any
from urllib.parse import unquote

import boto3

from config import get_settings

LOG_GROUP = "/aws/omics/WorkflowLog"
RUN_ID_RE = re.compile(r"^\d{1,20}$")
_URL_RUN_RE = re.compile(r"/log-events/run(?:%2F|/)(\d+)")
MAX_LOG_LINES = 25
MAX_LINE_CHARS = 300
MAX_TOTAL_CHARS = 3500


def omics_run_id_from_wes(run: dict) -> str | None:
    """The HealthOmics run id, read from the log links WES stored in the run's outputs."""
    urls = (run.get("outputs") or {}).get("log_urls") or {}
    candidates = [urls.get("run_log")]
    task_logs = urls.get("task_logs") or {}
    if isinstance(task_logs, dict):
        candidates.extend(task_logs.values())
    for url in candidates:
        if isinstance(url, str):
            match = _URL_RUN_RE.search(url)
            if match:
                return match.group(1)
    return None


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe(omics_run_id: str, region: str) -> dict[str, Any]:
    if not RUN_ID_RE.match(omics_run_id):
        raise ValueError("not a HealthOmics run id")
    omics = boto3.client("omics", region_name=region)
    run = omics.get_run(id=omics_run_id)
    info: dict[str, Any] = {
        "run_id": omics_run_id,
        "name": run.get("name"),
        "status": run.get("status"),
        "failure_reason": run.get("failureReason"),
        "message": _clip(run.get("statusMessage") or "", 600),
        "workflow_id": run.get("workflowId"),
        "failed_tasks": [],
        "log_tail": [],
    }
    try:
        tasks = omics.list_run_tasks(id=omics_run_id).get("items", [])
        info["failed_tasks"] = [
            {"name": t.get("name"), "status": t.get("status"), "message": _clip(t.get("statusMessage") or "", 300)}
            for t in tasks
            if str(t.get("status", "")).upper() in ("FAILED", "CANCELLED")
        ][:5]
    except Exception:  # noqa: BLE001 - task detail is optional
        pass
    logs = boto3.client("logs", region_name=region)
    used = 0
    for stream in (f"run/{omics_run_id}/engine", f"run/{omics_run_id}"):
        try:
            events = logs.get_log_events(
                logGroupName=LOG_GROUP, logStreamName=stream, limit=MAX_LOG_LINES, startFromHead=False
            ).get("events", [])
        except Exception:  # noqa: BLE001 - the stream may not exist or may be empty
            continue
        for event in events:
            line = _clip(unquote(event.get("message", "")), MAX_LINE_CHARS)
            if line and used + len(line) <= MAX_TOTAL_CHARS:
                info["log_tail"].append(line)
                used += len(line)
        if info["log_tail"]:
            break
    return info


async def describe_failure(omics_run_id: str) -> dict[str, Any]:
    return await asyncio.to_thread(_describe, omics_run_id, get_settings().AWS_REGION)
