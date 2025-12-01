#!/usr/bin/env python
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http import StreamableHTTPServerTransport
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

# Pagination defaults and limits (configurable via environment variables)
DEFAULT_SILENCE_PAGE = int(os.environ.get("ALERTMANAGER_DEFAULT_SILENCE_PAGE", "10"))
MAX_SILENCE_PAGE = int(os.environ.get("ALERTMANAGER_MAX_SILENCE_PAGE", "50"))
DEFAULT_ALERT_PAGE = int(os.environ.get("ALERTMANAGER_DEFAULT_ALERT_PAGE", "10"))
MAX_ALERT_PAGE = int(os.environ.get("ALERTMANAGER_MAX_ALERT_PAGE", "25"))
DEFAULT_ALERT_GROUP_PAGE = int(os.environ.get("ALERTMANAGER_DEFAULT_ALERT_GROUP_PAGE", "3"))
MAX_ALERT_GROUP_PAGE = int(os.environ.get("ALERTMANAGER_MAX_ALERT_GROUP_PAGE", "5"))


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
        Arbitrary keyword arguments.


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
        method=method.upper(), url=route, auth=auth, timeout=60, **kwargs
    )
    response.raise_for_status()
    return response.json()


def validate_pagination_params(count: int, offset: int, max_count: int) -> tuple[int, int, Optional[str]]:
    """Validate and normalize pagination parameters.

    Parameters
    ----------
    count : int
        Requested number of items per page
    offset : int
        Requested offset for pagination
    max_count : int
        Maximum allowed count value

    Returns
    -------
    tuple[int, int, Optional[str]]
        A tuple of (normalized_count, normalized_offset, error_message).
        If error_message is not None, the parameters are invalid and should
        return an error to the caller.
    """
    error = None

    # Validate count parameter
    if count < 1:
        error = f"Count parameter ({count}) must be at least 1."
    elif count > max_count:
        error = (
            f"Count parameter ({count}) exceeds maximum allowed value ({max_count}). "
            f"Please use count <= {max_count} and paginate through results using the offset parameter."
        )

    # Validate offset parameter
    if offset < 0:
        error = f"Offset parameter ({offset}) must be non-negative (>= 0)."

    return count, offset, error


def paginate_results(items: List[Any], count: int, offset: int) -> Dict[str, Any]:
    """Apply pagination to a list of items and generate pagination metadata.

    Parameters
    ----------
    items : List[Any]
        The full list of items to paginate
    count : int
        Number of items to return per page (must be >= 1)
    offset : int
        Number of items to skip (must be >= 0)

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - data: List of items for the current page
        - pagination: Metadata including total, offset, count, requested_count, and has_more
    """
    total = len(items)
    end_index = offset + count
    paginated_items = items[offset:end_index]
    has_more = end_index < total

    return {
        "data": paginated_items,
        "pagination": {
            "total": total,
            "offset": offset,
            "count": len(paginated_items),
            "requested_count": count,
            "has_more": has_more
        }
    }


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
async def get_silences(filter: Optional[str] = None,
                       count: int = DEFAULT_SILENCE_PAGE,
                       offset: int = 0):
    """Get list of all silences

    Parameters
    ----------
    filter
        Filtering query (e.g. alertname=~'.*CPU.*')"),
    count
        Number of silences to return per page (default: 10, max: 50).
    offset
        Number of silences to skip before returning results (default: 0).
        To paginate through all results, make multiple calls with increasing
        offset values (e.g., offset=0, offset=10, offset=20, etc.).

    Returns
    -------
    dict
        A dictionary containing:
        - data: List of Silence objects for the current page
        - pagination: Metadata about pagination (total, offset, count, has_more)
          Use the 'has_more' flag to determine if additional pages are available.
    """
    # Validate pagination parameters
    count, offset, error = validate_pagination_params(count, offset, MAX_SILENCE_PAGE)
    if error:
        return {"error": error}

    params = None
    if filter:
        params = {"filter": filter}

    # Get all silences from the API
    all_silences = make_request(method="GET", route="/api/v2/silences", params=params)

    # Apply pagination and return results
    return paginate_results(all_silences, count, offset)


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
                     active: Optional[bool] = None,
                     count: int = DEFAULT_ALERT_PAGE,
                     offset: int = 0):
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
    count
        Number of alerts to return per page (default: 10, max: 25).
    offset
        Number of alerts to skip before returning results (default: 0).
        To paginate through all results, make multiple calls with increasing
        offset values (e.g., offset=0, offset=10, offset=20, etc.).

    Returns
    -------
    dict
        A dictionary containing:
        - data: List of Alert objects for the current page
        - pagination: Metadata about pagination (total, offset, count, has_more)
          Use the 'has_more' flag to determine if additional pages are available.
    """
    # Validate pagination parameters
    count, offset, error = validate_pagination_params(count, offset, MAX_ALERT_PAGE)
    if error:
        return {"error": error}

    params = {"active": True}
    if filter:
        params = {"filter": filter}
    if silenced is not None:
        params["silenced"] = silenced
    if inhibited is not None:
        params["inhibited"] = inhibited
    if active is not None:
        params["active"] = active

    # Get all alerts from the API
    all_alerts = make_request(method="GET", route="/api/v2/alerts", params=params)

    # Apply pagination and return results
    return paginate_results(all_alerts, count, offset)


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
                           active: Optional[bool] = None,
                           count: int = DEFAULT_ALERT_GROUP_PAGE,
                           offset: int = 0):
    """Get a list of alert groups

    Params
    ------
    silenced
        If true, include silenced alerts.
    inhibited
        If true, include inhibited alerts.
    active
        If true, include active alerts.
    count
        Number of alert groups to return per page (default: 3, max: 5).
        Alert groups can be large as they contain all alerts within the group.
    offset
        Number of alert groups to skip before returning results (default: 0).
        To paginate through all results, make multiple calls with increasing
        offset values (e.g., offset=0, offset=3, offset=6, etc.).

    Returns
    -------
    dict
        A dictionary containing:
        - data: List of AlertGroup objects for the current page
        - pagination: Metadata about pagination (total, offset, count, has_more)
          Use the 'has_more' flag to determine if additional pages are available.
    """
    # Validate pagination parameters
    count, offset, error = validate_pagination_params(count, offset, MAX_ALERT_GROUP_PAGE)
    if error:
        return {"error": error}

    params = {"active": True}
    if silenced is not None:
        params["silenced"] = silenced
    if inhibited is not None:
        params["inhibited"] = inhibited
    if active is not None:
        params["active"] = active

    # Get all alert groups from the API
    all_groups = make_request(method="GET", route="/api/v2/alerts/groups",
                              params=params)

    # Apply pagination and return results
    return paginate_results(all_groups, count, offset)


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


def create_streamable_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that serves the Streamable HTTP transport.

    This starts the MCP server inside an application startup task using the
    transport.connect() context manager so the transport's in-memory streams
    are connected to the MCP server. The transport's ASGI handler is mounted
    at the '/mcp' path for GET/POST/DELETE requests.
    """
    transport = StreamableHTTPServerTransport(None)

    routes = [
        Mount("/mcp", app=transport.handle_request),
    ]

    app = Starlette(debug=debug, routes=routes)

    async def _startup() -> None:
        # Run the MCP server in a background asyncio task so the lifespan
        # event doesn't block. Store the task on app.state so shutdown can
        # cancel it.
        import asyncio

        async def _run_mcp() -> None:
            # Create the transport-backed streams and run the MCP server
            async with transport.connect() as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream, write_stream, mcp_server.create_initialization_options()
                )

        app.state._mcp_task = asyncio.create_task(_run_mcp())

    async def _shutdown() -> None:
        task = getattr(app.state, "_mcp_task", None)
        if task:
            task.cancel()
            try:
                await task
            except Exception:
                # Task cancelled or errored during shutdown is fine
                pass

        # Attempt to terminate the transport cleanly
        try:
            await transport.terminate()
        except Exception:
            pass

    app.add_event_handler("startup", _startup)
    app.add_event_handler("shutdown", _shutdown)

    return app


def run_server():
    """Main entry point for the Prometheus Alertmanager MCP Server"""
    setup_environment()
    # Get the underlying MCP server from the FastMCP instance
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Run MCP server with configurable transport')

    # Allow configuring defaults from environment variables. CLI arguments
    # (when provided) will override these environment values.
    env_transport = os.environ.get("MCP_TRANSPORT")
    env_host = os.environ.get("MCP_HOST")
    env_port = os.environ.get("MCP_PORT")

    transport_default = env_transport if env_transport is not None else 'stdio'
    host_default = env_host if env_host is not None else '0.0.0.0'
    try:
        port_default = int(env_port) if env_port is not None else 8000
    except (TypeError, ValueError):
        print(f"Invalid MCP_PORT value '{env_port}', falling back to 8000")
        port_default = 8000

    # Allow choosing between stdio and SSE transport modes
    parser.add_argument('--transport', choices=['stdio', 'http', 'sse'], default=transport_default,
                        help='Transport mode (stdio, http or sse) — can also be set via $MCP_TRANSPORT')
    # Host configuration for SSE mode
    parser.add_argument('--host', default=host_default,
                        help='Host to bind to (for SSE mode) — can also be set via $MCP_HOST')
    # Port configuration for SSE mode
    parser.add_argument('--port', type=int, default=port_default,
                        help='Port to listen on (for SSE mode) — can also be set via $MCP_PORT')
    args = parser.parse_args()
    print("\nStarting Prometheus Alertmanager MCP Server...")

    # Launch the server with the selected transport mode
    if args.transport == 'sse':
        print("Running server with SSE transport (web-based)")
        # Run with SSE transport (web-based)
        # Create a Starlette app to serve the MCP server
        starlette_app = create_starlette_app(mcp_server, debug=True)
        # Start the web server with the configured host and port
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    elif args.transport == 'http':
        print("Running server with http transport (streamable HTTP)")
        # Run with streamable-http transport served by uvicorn so host/port
        # CLI/env variables control the listening socket (same pattern as SSE).
        starlette_app = create_streamable_app(mcp_server, debug=True)
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        print("Running server with stdio transport (default)")
        # Run with stdio transport (default)
        # This mode communicates through standard input/output
        mcp.run(transport='stdio')


if __name__ == "__main__":
    run_server()
