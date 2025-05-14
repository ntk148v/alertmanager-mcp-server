#!/usr/bin/env python
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import sys

import dotenv
import requests
from mcp.server.fastmcp import FastMCP
from requests.compat import urljoin

dotenv.load_dotenv()
mcp = FastMCP("Alertmanager MCP")


@dataclass
class AlertmanagerConfig:
    url: str
    # Optional credentials
    username: Optional[str] = None
    password: Optional[str] = None


config = AlertmanagerConfig(
    url=os.environ.get("ALERTMANAGER_URL", ""),
    username=os.environ.get("ALERTMANAGER_USERNAME", ""),
    password=os.environ.get("ALERTMANAGER_PASSWORD", ""),
)


def make_request(method="GET", route="/", **kwargs):
    """Make HTTP request and return a requests.Response object.

    Parameters
    ----------
    method : str
        HTTP method to use for the request.
    route : str
        (Default value = "/")
        This is the url we are making our request to.
    **kwargs : dict
        Arbitrary keyword arguments.s


    Returns
    -------
    dict:
        The response from the Alertmanager API. This is a dictionary
        containing the response data.
    """
    route = urljoin(config.url, route)
    auth = (
        requests.auth.HTTPBasicAuth(config.username, config.password)
        if config.username and config.password
        else None
    )
    response = requests.request(
        method=method.upper(), url=route, auth=auth, **kwargs
    )
    response.raise_for_status()
    return response.json()


@mcp.tool(description="Get current status of an Alertmanager instance and its cluster")
async def get_status():
    """Get current status of an Alertmanager instance and its cluster

    Returns
    -------
    dict:
        The response from the Alertmanager API. This is a dictionary
        containing the response data.
    """
    return make_request(method="GET", route="/api/v2/status")


@mcp.tool(description="Get list of all receivers (name of notification integrations)")
async def get_receivers():
    """Get list of all receivers (name of notification integrations)

    Returns
    -------
    list:
        Return a list of Receiver objects from Alertmanager instance.
    """
    return make_request(method="GET", route="/api/v2/receivers")


@mcp.tool(description="Get list of all silences")
async def get_silences(filter: Optional[str] = None):
    """Get list of all silences

    Parameters
    ----------
    filter
        Filtering query (e.g. alertname=~'.*CPU.*')"),

    Returns
    -------
    list:
        Return a list of Silence objects from Alertmanager instance.
    """

    params = None
    if filter:
        params = {"filter": filter}
    return make_request(method="GET", route="/api/v2/silences", params=params)


@mcp.tool(description="Post a new silence or update an existing one")
async def post_silence(silence: Dict[str, Any]):
    """Post a new silence or update an existing one

    Parameters
    ----------
    silence : dict
        A dict representing the silence to be posted. This dict should
        contain the following keys:
            - matchers: list of matchers to match alerts to silence
            - startsAt: start time of the silence
            - endsAt: end time of the silence
            - createdBy: name of the user creating the silence
            - comment: comment for the silence

    Returns
    -------
    dict:
        Create / update silence response from Alertmanager API.
    """
    return make_request(method="POST", route="/api/v2/silences", json=silence)


@mcp.tool(description="Get a silence by its ID")
async def get_silence(silence_id: str):
    """Get a silence by its ID

    Parameters
    ----------
    silence_id : str
        The ID of the silence to be retrieved.

    Returns
    -------
    dict:
        The Silence object from Alertmanager instance.
    """
    return make_request(method="GET", route=urljoin("/api/v2/silences/", silence_id))


@mcp.tool(description="Delete a silence by its ID")
async def delete_silence(silence_id: str):
    """Delete a silence by its ID

    Parameters
    ----------
    silence_id : str
        The ID of the silence to be deleted.

    Returns
    -------
    dict:
        The response from the Alertmanager API.
    """
    return make_request(
        method="DELETE", route=urljoin("/api/v2/silences/", silence_id)
    )


@mcp.tool(description="Get a list of alerts")
async def get_alerts(filter: Optional[str] = None,
                     silenced: Optional[bool] = None,
                     inhibited: Optional[bool] = None,
                     active: Optional[bool] = None):
    """Get a list of alerts currently in Alertmanager.

    Params
    ------
    filter
        Filtering query (e.g. alertname=~'.*CPU.*')"),
    silenced
        If true, include silenced alerts.
    inhibited
        If true, include inhibited alerts.
    active
        If true, include active alerts.

    Returns
    -------
    list
        Return a list of Alert objects from Alertmanager instance.
    """
    params = {"active": True}
    if filter:
        params = {"filter": filter}
    if silenced is not None:
        params["silenced"] = silenced
    if inhibited is not None:
        params["inhibited"] = inhibited
    if active is not None:
        params["active"] = active
    return make_request(method="GET", route="/api/v2/alerts", params=params)


@mcp.tool(description="Create new alerts")
async def post_alerts(alerts: List[Dict]):
    """Create new alerts

    Parameters
    ----------
    alerts
        A list of Alert object.
        [
            {
                "startsAt": datetime,
                "endsAt": datetime,
                "annotations": labelSet
            }
        ]

    Returns
    -------
    dict:
        Create alert response from Alertmanager API.
    """
    return make_request(method="POST", route="/api/v2/alerts", json=alerts)


@mcp.tool(description="Get a list of alert groups")
async def get_alert_groups(silenced: Optional[bool] = None,
                           inhibited: Optional[bool] = None,
                           active: Optional[bool] = None):
    """Get a list of alert groups

    Params
    ------
    silenced
        If true, include silenced alerts.
    inhibited
        If true, include inhibited alerts.
    active
        If true, include active alerts.

    Returns
    -------
    list
        Return a list of AlertGroup objects from Alertmanager instance.
    """
    params = {"active": True}
    if silenced is not None:
        params["silenced"] = silenced
    if inhibited is not None:
        params["inhibited"] = inhibited
    if active is not None:
        params["active"] = active
    return make_request(method="GET", route="/api/v2/alerts/groups",
                        params=params)


def setup_environment():
    if dotenv.load_dotenv():
        print("Loaded environment variables from .env file")
    else:
        print("No .env file found or could not load it - using environment variables")

    if not config.url:
        print("ERROR: ALERTMANAGER_URL environment variable is not set")
        print("Please set it to your Alertmanager server URL")
        print("Example: http://your-alertmanager:9093")
        return False

    print("Alertmanager configuration:")
    print(f"  Server URL: {config.url}")

    if config.username and config.password:
        print("  Authentication: Using basic auth")
    else:
        print("  Authentication: None (no credentials provided)")

    return True


def run_server():
    """Main entry point for the Prometheus Alertmanager MCP Server"""
    # Setup environment
    if not setup_environment():
        sys.exit(1)

    print("\nStarting Prometheus Alertmanager MCP Server...")
    print("Running server in standard mode...")

    # Run the server with the stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
