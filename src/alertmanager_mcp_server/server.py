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


def _validate_get_alert_kwargs(**kwargs):
    """
    Check kwargs for validity.

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments. These kwargs are used to specify
        filters to limit the return of our list of alerts to alerts that
        match our filter.


    Raises
    ------
    KeyError
        If a key in our kwargs doesn't match our list of valid_keys,
        we raise a key error. We prevent filter keys that Alert Manager
        doesn't understand from being passed in a request.

    """
    valid_keys = ["filter", "silenced", "inhibited", "receiver"]
    for key in kwargs.keys():
        if key not in valid_keys:
            raise KeyError("invalid get parameter {}".format(key))


def _validate_get_silence_kwargs(**kwargs):
    """
    Check kwargs for validity.

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments. These kwargs are used to specify
        filters to limit the return of our list of silences to silences that
        match our filter.


    Raises
    ------
    KeyError
        If a key in our kwargs doesn't match our list of valid_keys,
        we raise a key error. We prevent filter keys that Alert Manager
        doesn't understand from being passed in a request.

    """
    valid_keys = ["filter"]
    for key in kwargs.keys():
        if key not in valid_keys:
            raise KeyError("invalid get parameter {}".format(key))


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


def _handle_filters(filter_dict):
    """
    Construct and return a filter.

    This is a protected method and should not be used outside of the public
    get_alerts method. This method works to ensure the structure of our
    filter string is something that Alert Manager can understand.

    Parameters
    ----------
    filter_dict : dict
        A dict where the keys represent the label on which we wish to
        filter and the value that key should have.


    Returns
    -------
    list
        Returns a list of filter strings to be passed along with our
        get_alerts method call.
    """
    if not isinstance(filter_dict, dict):
        raise TypeError("get_alerts() and get_silences() filter must be dict")
    filter_list = list()
    starter_string = '{}="{}"'
    for key, value in filter_dict.items():
        string = starter_string.format(key, value)
        filter_list.append(string)
    return filter_list


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
async def get_silences(**kwargs):
    """Get list of all silences

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments. These kwargs can be used to specify
        filters to limit the return of our list of alerts to silences that
        match our filter.


    Returns
    -------
    list:
        Return a list of Silence objects from Alertmanager instance.
    """

    _validate_get_silence_kwargs(**kwargs)
    if kwargs.get("filter"):
        kwargs["filter"] = _handle_filters(kwargs["filter"])
    return make_request(method="GET", route="/api/v2/silences")


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
async def get_alerts(**kwargs):
    """Get a list of alerts currently in Alertmanager.

    Params
    ------
    **kwargs : dict
        Arbitrary keyword arguments. These kwargs can be used to specify
        filters to limit the return of our list of alerts to alerts that
        match our filter.

    Returns
    -------
    list
        Return a list of Alert objects from Alertmanager instance.
    """
    _validate_get_alert_kwargs(**kwargs)
    if kwargs.get("filter"):
        kwargs["filter"] = _handle_filters(kwargs["filter"])
    return make_request(method="GET", route="/api/v2/alerts", params=kwargs)


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
async def get_alert_groups(**kwargs):
    """Get a list of alert groups

    Params
    ------
    **kwargs : dict
        Arbitrary keyword arguments. These kwargs can be used to specify
        filters to limit the return of our list of alerts to alerts that
        match our filter.

    Returns
    -------
    list
        Return a list of AlertGroup objects from Alertmanager instance.
    """
    _validate_get_alert_kwargs(**kwargs)
    if kwargs.get("filter"):
        kwargs["filter"] = _handle_filters(kwargs["filter"])
    return make_request(method="GET", route="/api/v2/alerts/groups",
                        params=kwargs)


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
        print("Authentication: Using basic auth")
    else:
        print("Authentication: None (no credentials provided)")

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
