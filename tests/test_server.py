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
    mock_silences = [
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
    mock_make_request.return_value = mock_silences
    # test with no filter
    result = await server.get_silences()
    mock_make_request.assert_called_with(
        method="GET", route="/api/v2/silences", params=None)
    # Check pagination structure
    assert "data" in result
    assert "pagination" in result
    assert result["data"][0]["status"]["state"] == "active"
    assert result["pagination"]["total"] == 1
    assert result["pagination"]["count"] == 1
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["has_more"] is False
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
    mock_alerts = [
        {
            "labels": {"alertname": "HighCPU"},
            "annotations": {"summary": "CPU usage high"},
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z",
            "status": {"state": "active"}
        }
    ]
    mock_make_request.return_value = mock_alerts
    # test with no params
    result = await server.get_alerts()
    mock_make_request.assert_any_call(
        method="GET", route="/api/v2/alerts", params={"active": True})
    # Check pagination structure
    assert "data" in result
    assert "pagination" in result
    assert result["data"][0]["labels"]["alertname"] == "HighCPU"
    assert result["pagination"]["total"] == 1
    assert result["pagination"]["count"] == 1
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["has_more"] is False
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
    mock_groups = [
        {
            "labels": {"severity": "critical"},
            "blocks": [],
            "alerts": [
                {"labels": {"alertname": "HighCPU"}, "status": {"state": "active"}}
            ]
        }
    ]
    mock_make_request.return_value = mock_groups
    # test with no params
    result = await server.get_alert_groups()
    mock_make_request.assert_any_call(
        method="GET", route="/api/v2/alerts/groups", params={"active": True})
    # Check pagination structure
    assert "data" in result
    assert "pagination" in result
    assert result["data"][0]["labels"]["severity"] == "critical"
    assert result["pagination"]["total"] == 1
    assert result["pagination"]["count"] == 1
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["has_more"] is False
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


@pytest.mark.asyncio
async def test_get_silences_pagination_default(mock_make_request):
    """Test get_silences with default pagination (10 items)"""
    # Create 25 mock silences
    mock_silences = [
        {
            "id": f"silence{i}",
            "status": {"state": "active"},
            "matchers": [{"name": "alertname", "value": f"Alert{i}", "isRegex": False}],
            "createdBy": "test",
            "comment": f"Silence {i}",
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z"
        }
        for i in range(25)
    ]
    mock_make_request.return_value = mock_silences

    # Test first page (default count=10, offset=0)
    result = await server.get_silences()
    assert len(result["data"]) == 10
    assert result["pagination"]["total"] == 25
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["count"] == 10
    assert result["pagination"]["requested_count"] == 10
    assert result["pagination"]["has_more"] is True
    assert result["data"][0]["id"] == "silence0"
    assert result["data"][9]["id"] == "silence9"


@pytest.mark.asyncio
async def test_get_silences_pagination_custom_count_offset(mock_make_request):
    """Test get_silences with custom count and offset"""
    # Create 25 mock silences
    mock_silences = [
        {
            "id": f"silence{i}",
            "status": {"state": "active"},
            "matchers": [{"name": "alertname", "value": f"Alert{i}", "isRegex": False}],
            "createdBy": "test",
            "comment": f"Silence {i}",
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z"
        }
        for i in range(25)
    ]
    mock_make_request.return_value = mock_silences

    # Test second page (count=10, offset=10)
    result = await server.get_silences(count=10, offset=10)
    assert len(result["data"]) == 10
    assert result["pagination"]["total"] == 25
    assert result["pagination"]["offset"] == 10
    assert result["pagination"]["count"] == 10
    assert result["pagination"]["has_more"] is True
    assert result["data"][0]["id"] == "silence10"
    assert result["data"][9]["id"] == "silence19"

    # Test last page (count=10, offset=20)
    result = await server.get_silences(count=10, offset=20)
    assert len(result["data"]) == 5  # Only 5 items left
    assert result["pagination"]["total"] == 25
    assert result["pagination"]["offset"] == 20
    assert result["pagination"]["count"] == 5
    assert result["pagination"]["has_more"] is False
    assert result["data"][0]["id"] == "silence20"
    assert result["data"][4]["id"] == "silence24"


@pytest.mark.asyncio
async def test_get_silences_pagination_max_count(mock_make_request):
    """Test get_silences with count exceeding maximum (50)"""
    # Request 100 items should return an error
    result = await server.get_silences(count=100)
    assert "error" in result
    assert "100" in result["error"]
    assert "50" in result["error"]
    assert "offset" in result["error"].lower()

    # Verify that requesting exactly at the limit works
    mock_silences = [
        {
            "id": f"silence{i}",
            "status": {"state": "active"},
            "matchers": [{"name": "alertname", "value": f"Alert{i}", "isRegex": False}],
            "createdBy": "test",
            "comment": f"Silence {i}",
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z"
        }
        for i in range(100)
    ]
    mock_make_request.return_value = mock_silences

    result = await server.get_silences(count=50)
    assert "error" not in result
    assert len(result["data"]) == 50
    assert result["pagination"]["total"] == 100


@pytest.mark.asyncio
async def test_get_silences_pagination_empty_results(mock_make_request):
    """Test get_silences with empty results"""
    mock_make_request.return_value = []

    result = await server.get_silences()
    assert len(result["data"]) == 0
    assert result["pagination"]["total"] == 0
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["count"] == 0
    assert result["pagination"]["has_more"] is False


@pytest.mark.asyncio
async def test_get_alerts_pagination_default(mock_make_request):
    """Test get_alerts with default pagination (10 items)"""
    # Create 25 mock alerts
    mock_alerts = [
        {
            "labels": {"alertname": f"Alert{i}"},
            "annotations": {"summary": f"Alert {i}"},
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z",
            "status": {"state": "active"}
        }
        for i in range(25)
    ]
    mock_make_request.return_value = mock_alerts

    # Test first page (default count=10, offset=0)
    result = await server.get_alerts()
    assert len(result["data"]) == 10
    assert result["pagination"]["total"] == 25
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["count"] == 10
    assert result["pagination"]["requested_count"] == 10
    assert result["pagination"]["has_more"] is True
    assert result["data"][0]["labels"]["alertname"] == "Alert0"
    assert result["data"][9]["labels"]["alertname"] == "Alert9"


@pytest.mark.asyncio
async def test_get_alerts_pagination_custom_count_offset(mock_make_request):
    """Test get_alerts with custom count and offset"""
    # Create 25 mock alerts
    mock_alerts = [
        {
            "labels": {"alertname": f"Alert{i}"},
            "annotations": {"summary": f"Alert {i}"},
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z",
            "status": {"state": "active"}
        }
        for i in range(25)
    ]
    mock_make_request.return_value = mock_alerts

    # Test second page (count=10, offset=10)
    result = await server.get_alerts(count=10, offset=10)
    assert len(result["data"]) == 10
    assert result["pagination"]["total"] == 25
    assert result["pagination"]["offset"] == 10
    assert result["pagination"]["count"] == 10
    assert result["pagination"]["has_more"] is True
    assert result["data"][0]["labels"]["alertname"] == "Alert10"
    assert result["data"][9]["labels"]["alertname"] == "Alert19"

    # Test last page (count=10, offset=20)
    result = await server.get_alerts(count=10, offset=20)
    assert len(result["data"]) == 5  # Only 5 items left
    assert result["pagination"]["total"] == 25
    assert result["pagination"]["offset"] == 20
    assert result["pagination"]["count"] == 5
    assert result["pagination"]["has_more"] is False
    assert result["data"][0]["labels"]["alertname"] == "Alert20"
    assert result["data"][4]["labels"]["alertname"] == "Alert24"


@pytest.mark.asyncio
async def test_get_alerts_pagination_max_count(mock_make_request):
    """Test get_alerts with count exceeding maximum (25)"""
    # Request 50 items should return an error
    result = await server.get_alerts(count=50)
    assert "error" in result
    assert "50" in result["error"]
    assert "25" in result["error"]
    assert "offset" in result["error"].lower()

    # Verify that requesting exactly at the limit works
    mock_alerts = [
        {
            "labels": {"alertname": f"Alert{i}"},
            "annotations": {"summary": f"Alert {i}"},
            "startsAt": "2025-05-14T00:00:00Z",
            "endsAt": "2025-05-15T00:00:00Z",
            "status": {"state": "active"}
        }
        for i in range(50)
    ]
    mock_make_request.return_value = mock_alerts

    result = await server.get_alerts(count=25)
    assert "error" not in result
    assert len(result["data"]) == 25
    assert result["pagination"]["total"] == 50


@pytest.mark.asyncio
async def test_get_alerts_pagination_empty_results(mock_make_request):
    """Test get_alerts with empty results"""
    mock_make_request.return_value = []

    result = await server.get_alerts()
    assert len(result["data"]) == 0
    assert result["pagination"]["total"] == 0
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["count"] == 0
    assert result["pagination"]["has_more"] is False


@pytest.mark.asyncio
async def test_get_alert_groups_pagination_default(mock_make_request):
    """Test get_alert_groups with default pagination (3 items)"""
    # Create 15 mock alert groups
    mock_groups = [
        {
            "labels": {"severity": f"severity{i}"},
            "blocks": [],
            "alerts": []
        }
        for i in range(15)
    ]
    mock_make_request.return_value = mock_groups

    # Test first page (default count=3, offset=0)
    result = await server.get_alert_groups()
    assert len(result["data"]) == 3
    assert result["pagination"]["total"] == 15
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["count"] == 3
    assert result["pagination"]["requested_count"] == 3
    assert result["pagination"]["has_more"] is True


@pytest.mark.asyncio
async def test_get_alert_groups_pagination_custom_count_offset(mock_make_request):
    """Test get_alert_groups with custom count and offset"""
    # Create 12 mock alert groups
    mock_groups = [
        {
            "labels": {"severity": f"severity{i}"},
            "blocks": [],
            "alerts": []
        }
        for i in range(12)
    ]
    mock_make_request.return_value = mock_groups

    # Test second page (count=3, offset=3)
    result = await server.get_alert_groups(count=3, offset=3)
    assert len(result["data"]) == 3
    assert result["pagination"]["total"] == 12
    assert result["pagination"]["offset"] == 3
    assert result["pagination"]["count"] == 3
    assert result["pagination"]["has_more"] is True

    # Test last page (count=3, offset=9)
    result = await server.get_alert_groups(count=3, offset=9)
    assert len(result["data"]) == 3
    assert result["pagination"]["total"] == 12
    assert result["pagination"]["offset"] == 9
    assert result["pagination"]["count"] == 3
    assert result["pagination"]["has_more"] is False


@pytest.mark.asyncio
async def test_get_alert_groups_pagination_max_count(mock_make_request):
    """Test get_alert_groups with count exceeding maximum (5)"""
    # Request 10 items should return an error
    result = await server.get_alert_groups(count=10)
    assert "error" in result
    assert "10" in result["error"]
    assert "5" in result["error"]
    assert "offset" in result["error"].lower()

    # Verify that requesting exactly at the limit works
    mock_groups = [
        {
            "labels": {"severity": f"severity{i}"},
            "blocks": [],
            "alerts": []
        }
        for i in range(15)
    ]
    mock_make_request.return_value = mock_groups

    result = await server.get_alert_groups(count=5)
    assert "error" not in result
    assert len(result["data"]) == 5
    assert result["pagination"]["total"] == 15


@patch("alertmanager_mcp_server.server.setup_environment", return_value=True)
@patch("alertmanager_mcp_server.server.mcp")
def test_run_server_success(mock_mcp, mock_setup_env):
    with patch("builtins.print") as mock_print, \
         patch("sys.argv", ["server.py"]):
        server.run_server()
        mock_setup_env.assert_called_once()
        mock_mcp.run.assert_called_once_with(transport="stdio")
        assert any("Starting Prometheus Alertmanager MCP Server" in str(call)
                   for call in mock_print.call_args_list)
