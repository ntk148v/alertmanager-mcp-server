#!/usr/bin/env python
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from requests.compat import urljoin
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import dotenv
import requests
import uvicorn

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


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided MCP server with SSE.

    Sets up a Starlette web application with routes for SSE (Server-Sent Events)
    communication with the MCP server.

    Args:
        mcp_server: The MCP server instance to connect
        debug: Whether to enable debug mode for the Starlette app

    Returns:
        A configured Starlette application
    """
    # Create an SSE transport with a base path for messages
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        """Handler for SSE connections.

        Establishes an SSE connection and connects it to the MCP server.

        Args:
            request: The incoming HTTP request
        """
        # Connect the SSE transport to the request
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            # Run the MCP server with the SSE streams
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # Create and return the Starlette application with routes
    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),  # Endpoint for SSE connections
            # Endpoint for posting messages
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def run_server():
    """Main entry point for the Prometheus Alertmanager MCP Server"""
    setup_environment()
    # Get the underlying MCP server from the FastMCP instance
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Run MCP server with configurable transport')
    # Allow choosing between stdio and SSE transport modes
    parser.add_argument('--transport', choices=['stdio', 'sse'], default='stdio',
                        help='Transport mode (stdio or sse)')
    # Host configuration for SSE mode
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to (for SSE mode)')
    # Port configuration for SSE mode
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to listen on (for SSE mode)')
    args = parser.parse_args()
    print("\nStarting Prometheus Alertmanager MCP Server...")

    # Launch the server with the selected transport mode
    if args.transport == 'stdio':
        print("Running server with stdio transport (default)")
        # Run with stdio transport (default)
        # This mode communicates through standard input/output
        mcp.run(transport='stdio')
    else:
        print("Running server with SSE transport (web-based)")
        # Run with SSE transport (web-based)
        # Create a Starlette app to serve the MCP server
        starlette_app = create_starlette_app(mcp_server, debug=True)
        # Start the web server with the configured host and port
        uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == "__main__":
    run_server()
