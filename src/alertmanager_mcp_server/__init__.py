"""Prometheus Alertmanager MCP Server.

A Model Context Protocol (MCP) server that enables AI assistants to query
and integrate with Promeheus Alertmanager.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("alertmanager_mcp_server")
except PackageNotFoundError:
    __version__ = "0.0.0"
