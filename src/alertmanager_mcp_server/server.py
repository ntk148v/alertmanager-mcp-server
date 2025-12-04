#!/usr/bin/env python
import os
import logging
import socket
import signal
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import dotenv
import requests
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_interrupt(signum, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def safe_print(text):
    # Don't print to stderr when running as MCP server via uvx to avoid JSON parsing errors
    # Check if we're running as MCP server (no TTY and uvx in process name)
    if not sys.stderr.isatty():
        # Running as MCP server, suppress output to avoid JSON parsing errors
        logger.debug(f"[MCP Server] {text}")
        return

    try:
        print(text, file=sys.stderr)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode(), file=sys.stderr)


def check_port(port):
    # Check port availability before starting HTTP server
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
    except OSError as e:
        safe_print(f"Socket error: {e}")
        safe_print(
            f"Port {port} is already in use. Cannot start HTTP server.")
        sys.exit(1)


dotenv.load_dotenv()
mcp = FastMCP("Alertmanager MCP")

# ContextVar for per-request X-Scope-OrgId header
# Used for multi-tenant Alertmanager setups (e.g., Mimir)
# ContextVar ensures proper isolation per async context/task
_current_scope_org_id: ContextVar[Optional[str]] = ContextVar(
    "current_scope_org_id", default=None)


def extract_header_from_scope(scope: dict, header_name: str) -> Optional[str]:
    """Extract a header value from an ASGI scope.

    Parameters
    ----------
    scope : dict
        ASGI scope dictionary containing headers
    header_name : str
        Header name to extract (should be lowercase, e.g. "x-scope-orgid")

    Returns
    -------
    Optional[str]
        The header value if found, None otherwise
    """
    headers = scope.get("headers", [])
    target = header_name.encode("latin-1")
    for name_bytes, value_bytes in headers:
        if name_bytes.lower() == target:
            try:
                return value_bytes.decode("latin-1")
            except Exception:
                return None
    return None


def extract_header_from_request(request: Request, header_name: str) -> Optional[str]:
    """Extract a header value from a Starlette Request.

    Parameters
    ----------
    request : Request
        Starlette request object
    header_name : str
        Header name to extract (case-insensitive)

    Returns
    -------
    Optional[str]
        The header value if found, None otherwise
    """
    return request.headers.get(header_name)


@dataclass
class AlertmanagerConfig:
    url: str
    # Optional credentials
    username: Optional[str] = None
    password: Optional[str] = None
    # Optional tenant ID for multi-tenant setups
    tenant_id: Optional[str] = None


config = AlertmanagerConfig(
    url=os.environ.get("ALERTMANAGER_URL", ""),
    username=os.environ.get("ALERTMANAGER_USERNAME", ""),
    password=os.environ.get("ALERTMANAGER_PASSWORD", ""),
    tenant_id=os.environ.get("ALERTMANAGER_TENANT", ""),
)

# Pagination defaults and limits (configurable via environment variables)
DEFAULT_SILENCE_PAGE = int(os.environ.get(
    "ALERTMANAGER_DEFAULT_SILENCE_PAGE", "10"))
MAX_SILENCE_PAGE = int(os.environ.get("ALERTMANAGER_MAX_SILENCE_PAGE", "50"))
DEFAULT_ALERT_PAGE = int(os.environ.get(
    "ALERTMANAGER_DEFAULT_ALERT_PAGE", "10"))
MAX_ALERT_PAGE = int(os.environ.get("ALERTMANAGER_MAX_ALERT_PAGE", "25"))
DEFAULT_ALERT_GROUP_PAGE = int(os.environ.get(
    "ALERTMANAGER_DEFAULT_ALERT_GROUP_PAGE", "3"))
MAX_ALERT_GROUP_PAGE = int(os.environ.get(
    "ALERTMANAGER_MAX_ALERT_GROUP_PAGE", "5"))


def url_join(base: str, path: str) -> str:
    """Join a base URL with a path, preserving the base URL's path component.

    Unlike urllib.parse.urljoin, this function preserves the path in the base URL
    when the path argument starts with '/'. This is useful for APIs hosted at
    subpaths (e.g., http://localhost:8080/alertmanager).

    Examples
    --------
    >>> url_join("http://localhost:8080/alertmanager", "/api/v2/alerts")
    'http://localhost:8080/alertmanager/api/v2/alerts'

    >>> url_join("http://localhost:8080/alertmanager/", "/api/v2/alerts")
    'http://localhost:8080/alertmanager/api/v2/alerts'

    >>> url_join("http://localhost:8080", "/api/v2/alerts")
    'http://localhost:8080/api/v2/alerts'

    Parameters
    ----------
    base : str
        The base URL which may include a path component
    path : str
        The path to append, which may or may not start with '/'

    Returns
    -------
    str
        The combined URL with both base path and appended path
    """
    # Remove trailing slash from base if present
    base = base.rstrip('/')

    # Remove leading slash from path if present
    path = path.lstrip('/')

    # Combine with a single slash
    return f"{base}/{path}"


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
    try:
        route = url_join(config.url, route)
        auth = (
            requests.auth.HTTPBasicAuth(config.username, config.password)
            if config.username and config.password
            else None
        )

        # Add X-Scope-OrgId header for multi-tenant setups
        # Priority: 1) Request header from caller (via ContextVar), 2) Static config tenant
        headers = kwargs.get("headers", {})

        tenant_id = _current_scope_org_id.get() or config.tenant_id

        if tenant_id:
            headers["X-Scope-OrgId"] = tenant_id
        if headers:
            kwargs["headers"] = headers

        response = requests.request(
            method=method.upper(), url=route, auth=auth, timeout=60, **kwargs
        )
        response.raise_for_status()
        result = response.json()

        # Ensure we always return something (empty list is valid but might cause issues)
        if result is None:
            return {"message": "No data returned"}
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


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
    count, offset, error = validate_pagination_params(
        count, offset, MAX_SILENCE_PAGE)
    if error:
        return {"error": error}

    params = None
    if filter:
        params = {"filter": filter}

    # Get all silences from the API
    all_silences = make_request(
        method="GET", route="/api/v2/silences", params=params)

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
    return make_request(method="GET", route=url_join("/api/v2/silences/", silence_id))


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
        method="DELETE", route=url_join("/api/v2/silences/", silence_id)
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
    count, offset, error = validate_pagination_params(
        count, offset, MAX_ALERT_PAGE)
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
    all_alerts = make_request(
        method="GET", route="/api/v2/alerts", params=params)

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
    count, offset, error = validate_pagination_params(
        count, offset, MAX_ALERT_GROUP_PAGE)
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
        safe_print("Loaded environment variables from .env file")
    else:
        safe_print(
            "No .env file found or could not load it - using environment variables")

    if not config.url:
        safe_print("ERROR: ALERTMANAGER_URL environment variable is not set")
        safe_print("Please set it to your Alertmanager server URL")
        safe_print("Example: http://your-alertmanager:9093")
        return False

    safe_print("Alertmanager configuration:")
    safe_print(f"  Server URL: {config.url}")

    if config.username and config.password:
        safe_print("  Authentication: Using basic auth")
    else:
        safe_print("  Authentication: None (no credentials provided)")

    if config.tenant_id:
        safe_print(f"  Static Tenant ID: {config.tenant_id}")
    else:
        safe_print("  Static Tenant ID: None")

    safe_print("\nMulti-tenant Support:")
    safe_print(
        "  - Send X-Scope-OrgId header with requests for multi-tenant setups")
    safe_print(
        "  - Request header takes precedence over static ALERTMANAGER_TENANT config")

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
        # Extract X-Scope-OrgId header if present and set in ContextVar
        scope_org_id = extract_header_from_request(request, "x-scope-orgid")
        token = _current_scope_org_id.set(
            scope_org_id) if scope_org_id else None

        try:
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
        finally:
            # Reset ContextVar to restore previous value
            if token is not None:
                _current_scope_org_id.reset(token)

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

    async def handle_mcp_request(scope, receive, send):
        """Wrapper to extract X-Scope-OrgId header before handling MCP request."""
        token = None

        if scope['type'] == 'http':
            # Extract X-Scope-OrgId from headers
            scope_org_id = extract_header_from_scope(scope, "x-scope-orgid")
            if scope_org_id:
                token = _current_scope_org_id.set(scope_org_id)

        try:
            # Pass to the actual transport handler
            await transport.handle_request(scope, receive, send)
        finally:
            # Reset ContextVar to restore previous value
            if token is not None:
                _current_scope_org_id.reset(token)

    routes = [
        Mount("/mcp", app=handle_mcp_request),
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
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
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
        safe_print(
            f"Invalid MCP_PORT value '{env_port}', falling back to 8000")
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
    safe_print("\nStarting Prometheus Alertmanager MCP Server...")

    # Launch the server with the selected transport mode
    if args.transport == 'sse':
        # Check port availability before starting HTTP server
        check_port(args.port)

        safe_print("Running server with SSE transport (web-based)")
        # Run with SSE transport (web-based)
        # Create a Starlette app to serve the MCP server
        starlette_app = create_starlette_app(mcp_server, debug=True)
        # Start the web server with the configured host and port
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    elif args.transport == 'http':
        # Check port availability before starting HTTP server
        check_port(args.port)

        safe_print("Running server with http transport (streamable HTTP)")
        # Run with streamable-http transport served by uvicorn so host/port
        # CLI/env variables control the listening socket (same pattern as SSE).
        starlette_app = create_streamable_app(mcp_server, debug=True)
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        safe_print("Running server with stdio transport (default)")
        # Run with stdio transport (default)
        # This mode communicates through standard input/output
        mcp.run(transport='stdio')


if __name__ == "__main__":
    run_server()
