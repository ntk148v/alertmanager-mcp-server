"""Tests for the Prometheus Alertmanager MCP server functionality."""

import importlib
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock

import alertmanager_mcp_server.server as server


@patch("alertmanager_mcp_server.server.requests.request")
def test_make_request_without_basic_auth_success(mock_request):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "success",
        "data": {
            "cluster": {"status": "ready"},
            "versionInfo": {"version": "0.28.0"}
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_request.return_value = mock_response
    result = server.make_request(method="GET", route="/api/v2/status")
    assert result == {
        "status": "success",
        "data": {
            "cluster": {"status": "ready"},
            "versionInfo": {"version": "0.28.0"}
        }
    }
    mock_request.assert_called_once()


@patch("alertmanager_mcp_server.server.requests.request")
def test_make_request_http_error(mock_request):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP error")
    mock_request.return_value = mock_response
    with pytest.raises(Exception):
        server.make_request(method="GET", route="/api/v2/status")


@patch("alertmanager_mcp_server.server.requests.request")
def test_make_request_with_basic_auth(mock_request):
    # Save original config
    server.config.username = "user"
    server.config.password = "pass"
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "success",
        "data": {
            "cluster": {"status": "ready"},
            "versionInfo": {"version": "0.28.0"}
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_request.return_value = mock_response
    server.make_request(method="GET", route="/api/v2/status")
    args, kwargs = mock_request.call_args
    assert kwargs["auth"] is not None


@pytest_asyncio.fixture
async def mock_make_request():
    with patch("alertmanager_mcp_server.server.make_request") as mock:
        yield mock


@pytest.mark.asyncio
async def test_get_status_tool(mock_make_request):
    # /status returns StatusResponse
    mock_make_request.return_value = {
        "status": "success",
        "data": {
            "cluster": {"status": "ready"},
            "versionInfo": {"version": "0.28.0"}
        }
    }
    result = await server.get_status()
    mock_make_request.assert_called_once_with(
        method="GET", route="/api/v2/status")
    assert result["status"] == "success"
    assert "cluster" in result["data"]


@pytest.mark.asyncio
async def test_get_receivers_tool(mock_make_request):
    # /receivers returns list of Receiver objects
    mock_make_request.return_value = [
        {"name": "slack", "email_configs": [], "webhook_configs": []},
        {"name": "pagerduty", "pagerduty_configs": []}
    ]
    result = await server.get_receivers()
    mock_make_request.assert_called_once_with(
        method="GET", route="/api/v2/receivers")
    assert isinstance(result, list)
    assert result[0]["name"] == "slack"


@pytest.mark.asyncio
async def test_get_silences_tool(mock_make_request):
    # /silences returns list of Silence objects, now supports filter param
    mock_make_request.return_value = [
        {
            "id": "123",
            "status": {"state": "active"},
            "matchers": [{"name": "alertname", "value": "HighCPU", "isRegex": False}],
            "createdBy": "me",
            "comment": "test",
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z"
        }
    ]
    # test with no filter
    result = await server.get_silences()
    mock_make_request.assert_called_with(
        method="GET", route="/api/v2/silences", params=None)
    assert result[0]["status"]["state"] == "active"
    # test with filter
    await server.get_silences(filter="alertname=HighCPU")
    mock_make_request.assert_called_with(
        method="GET", route="/api/v2/silences", params={"filter": "alertname=HighCPU"})


@pytest.mark.asyncio
async def test_post_silence_tool(mock_make_request):
    silence = {
        "matchers": [{"name": "alertname", "value": "HighCPU", "isRegex": False}],
        "startsAt": "2025-05-14T00:00:00Z",
        "endsAt": "2025-05-15T00:00:00Z",
        "createdBy": "me",
        "comment": "test"
    }
    # POST /silences returns { silenceID: string }
    mock_make_request.return_value = {"silenceID": "abc123"}
    result = await server.post_silence(silence)
    mock_make_request.assert_called_once_with(
        method="POST", route="/api/v2/silences", json=silence)
    assert result["silenceID"] == "abc123"


@pytest.mark.asyncio
async def test_get_silence_tool(mock_make_request):
    # /silences/{id} returns Silence object
    mock_make_request.return_value = {
        "id": "abc123",
        "status": {"state": "active"},
        "matchers": [{"name": "alertname", "value": "HighCPU", "isRegex": False}],
        "createdBy": "me",
        "comment": "test",
        "startsAt": "2025-05-14T00:00:00Z",
        "endsAt": "2025-05-15T00:00:00Z"
    }
    result = await server.get_silence("abc123")
    assert result["id"] == "abc123"
    assert result["status"]["state"] == "active"


@pytest.mark.asyncio
async def test_delete_silence_tool(mock_make_request):
    # DELETE /silences/{id} returns empty object
    mock_make_request.return_value = {}
    result = await server.delete_silence("abc123")
    assert result == {}


@pytest.mark.asyncio
async def test_get_alerts_tool(mock_make_request):
    # /alerts returns list of Alert objects, now supports filter, silenced, inhibited, active
    mock_make_request.return_value = [
        {
            "labels": {"alertname": "HighCPU"},
            "annotations": {"summary": "CPU usage high"},
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z",
            "status": {"state": "active"}
        }
    ]
    # test with no params
    result = await server.get_alerts()
    mock_make_request.assert_any_call(
        method="GET", route="/api/v2/alerts", params={"active": True})
    assert result[0]["labels"]["alertname"] == "HighCPU"
    # test with filter and flags
    await server.get_alerts(filter="alertname=HighCPU", silenced=True, inhibited=False, active=True)
    mock_make_request.assert_any_call(method="GET", route="/api/v2/alerts", params={
                                      "filter": "alertname=HighCPU", "silenced": True, "inhibited": False, "active": True})


@pytest.mark.asyncio
async def test_post_alerts_tool(mock_make_request):
    alerts = [
        {
            "labels": {"alertname": "HighCPU"},
            "annotations": {"summary": "CPU usage high"},
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z"
        }
    ]
    # POST /alerts returns empty object
    mock_make_request.return_value = {}
    result = await server.post_alerts(alerts)
    mock_make_request.assert_called_once_with(
        method="POST", route="/api/v2/alerts", json=alerts)
    assert result == {}


@pytest.mark.asyncio
async def test_get_alert_groups_tool(mock_make_request):
    # /alerts/groups returns list of AlertGroup objects, now supports silenced, inhibited, active
    mock_make_request.return_value = [
        {
            "labels": {"severity": "critical"},
            "blocks": [],
            "alerts": [
                {"labels": {"alertname": "HighCPU"}, "status": {"state": "active"}}
            ]
        }
    ]
    # test with no params
    result = await server.get_alert_groups()
    mock_make_request.assert_any_call(
        method="GET", route="/api/v2/alerts/groups", params={"active": True})
    assert result[0]["labels"]["severity"] == "critical"
    # test with flags
    await server.get_alert_groups(silenced=True, inhibited=True, active=False)
    mock_make_request.assert_any_call(method="GET", route="/api/v2/alerts/groups", params={
                                      "active": False, "silenced": True, "inhibited": True})


def test_setup_environment_with_basic_auth(monkeypatch):
    monkeypatch.setenv("ALERTMANAGER_URL", "http://localhost:9093")
    monkeypatch.setenv("ALERTMANAGER_USERNAME", "user")
    monkeypatch.setenv("ALERTMANAGER_PASSWORD", "pass")
    importlib.reload(server)
    with patch("builtins.print") as mock_print:
        assert server.setup_environment() is True
        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Authentication: Using basic auth" in output


def test_setup_environment_without_basic_auth(monkeypatch):
    monkeypatch.setenv("ALERTMANAGER_URL", "http://localhost:9093")
    monkeypatch.delenv("ALERTMANAGER_USERNAME", raising=False)
    monkeypatch.delenv("ALERTMANAGER_PASSWORD", raising=False)
    importlib.reload(server)
    with patch("builtins.print") as mock_print:
        assert server.setup_environment() is True
        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Authentication: None (no credentials provided)" in output


def test_setup_environment_no_url(monkeypatch):
    monkeypatch.delenv("ALERTMANAGER_URL", raising=False)
    # Reload config
    importlib.reload(server)
    assert server.setup_environment() is False


@patch("alertmanager_mcp_server.server.setup_environment", return_value=True)
@patch("alertmanager_mcp_server.server.mcp")
def test_run_server_success(mock_mcp, mock_setup_env):
    with patch("builtins.print") as mock_print:
        server.run_server()
        mock_setup_env.assert_called_once()
        mock_mcp.run.assert_called_once_with(transport="stdio")
        assert any("Starting Prometheus Alertmanager MCP Server" in str(call)
                   for call in mock_print.call_args_list)
