"""HealthOmics failure details: run id extraction, input validation, and bounded output."""

import asyncio

import pytest

import omics_client


def test_run_id_comes_from_the_wes_log_links():
    run = {"outputs": {"log_urls": {"run_log": "https://c/cloudwatch#logs/log-events/run%2F7408798"}}}
    assert omics_client.omics_run_id_from_wes(run) == "7408798"
    task_only = {"outputs": {"log_urls": {"task_logs": {"main.cwl": "https://c/x/log-events/run%2F42%2Ftask%2F9"}}}}
    assert omics_client.omics_run_id_from_wes(task_only) == "42"
    assert omics_client.omics_run_id_from_wes({"outputs": {}}) is None
    assert omics_client.omics_run_id_from_wes({}) is None
    assert omics_client.omics_run_id_from_wes({"outputs": {"log_urls": {"run_log": "https://c/other"}}}) is None


def test_only_numeric_ids_are_accepted():
    for bad in ("../x", "123/engine", "abc", "", "1" * 30, "12 34"):
        with pytest.raises(ValueError):
            omics_client._describe(bad, "us-east-1")


class _Omics:
    def __init__(self):
        self.calls = []

    def get_run(self, id):
        self.calls.append(("get_run", id))
        return {"name": "r", "status": "FAILED", "failureReason": "CANNOT_START_CONTAINER_ERROR",
                "statusMessage": "m " * 500, "workflowId": "7408931"}

    def list_run_tasks(self, id):
        self.calls.append(("list_run_tasks", id))
        return {"items": [{"name": "main.cwl", "status": "FAILED", "statusMessage": "x"},
                          {"name": "ok", "status": "COMPLETED"}]}


class _Logs:
    def __init__(self):
        self.streams = []

    def get_log_events(self, logGroupName, logStreamName, limit, startFromHead):
        self.streams.append((logGroupName, logStreamName, limit, startFromHead))
        if logStreamName.endswith("/engine"):
            return {"events": [{"message": "line %d " % i + "z" * 1000} for i in range(40)]}
        raise AssertionError("fell back although the engine stream had content")


def test_describe_reads_only_this_run_and_bounds_the_output(monkeypatch):
    omics, logs = _Omics(), _Logs()
    monkeypatch.setattr(omics_client.boto3, "client", lambda name, region_name=None: omics if name == "omics" else logs)
    info = omics_client._describe("2933968", "us-east-1")
    assert info["failure_reason"] == "CANNOT_START_CONTAINER_ERROR"
    assert [t["name"] for t in info["failed_tasks"]] == ["main.cwl"]  # only failed tasks
    assert all(c[1] == "2933968" for c in omics.calls)
    assert logs.streams == [("/aws/omics/WorkflowLog", "run/2933968/engine", omics_client.MAX_LOG_LINES, False)]
    assert len(info["message"]) <= 600
    assert all(len(line) <= omics_client.MAX_LINE_CHARS for line in info["log_tail"])
    assert sum(len(line) for line in info["log_tail"]) <= omics_client.MAX_TOTAL_CHARS


def test_describe_failure_is_async_and_uses_the_configured_region(monkeypatch):
    seen = {}
    monkeypatch.setattr(omics_client, "_describe", lambda rid, region: seen.update(rid=rid, region=region) or {"ok": 1})
    assert asyncio.run(omics_client.describe_failure("123")) == {"ok": 1}
    assert seen["rid"] == "123" and seen["region"]
