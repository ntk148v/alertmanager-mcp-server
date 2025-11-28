<div align="center">
    <h1>Prometheus Alertmanager MCP</h1>
    <p>
        <a href="https://github.com/ntk148v/alertmanager-mcp-server/blob/master/LICENSE">
            <img alt="GitHub license" src="https://img.shields.io/github/license/ntk148v/alertmanager-mcp-server?style=for-the-badge">
        </a>
    <a href="https://github.com/ntk148v/alertmanager-mcp-server/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/ntk148v/alertmanager-mcp-server?style=for-the-badge">
    </a>
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [1. Introduction](#1-introduction)
- [2. Features](#2-features)
- [3. Quickstart](#3-quickstart)
  - [3.1. Prerequisites](#31-prerequisites)
  - [3.2. Installing via Smithery](#32-installing-via-smithery)
  - [3.3. Local Run](#33-local-run)
  - [3.4. Docker Run](#34-docker-run)
- [4. Tools](#4-tools)
- [5. Development](#5-development)
- [6. License](#6-license)

## 1. Introduction

Prometheus Alertmanager MCP is a [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server for Prometheus Alertmanager. It enables AI assistants and tools to query and manage Alertmanager resources programmatically and securely.

## 2. Features

- [x] Query Alertmanager status, alerts, silences, receivers, and alert groups
- [x] **Smart pagination support** to prevent LLM context window overflow when handling large numbers of alerts
- [x] Create, update, and delete silences
- [x] Create new alerts
- [x] Authentication support (Basic auth via environment variables)
- [x] Docker containerization support

## 3. Quickstart

### 3.1. Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for fast dependency management).
- Docker (optional, for containerized deployment).
- Ensure your Prometheus Alertmanager server is accessible from the environment where you'll run this MCP server.

### 3.2. Installing via Smithery

To install Prometheus Alertmanager MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@ntk148v/alertmanager-mcp-server):

```bash
npx -y @smithery/cli install @ntk148v/alertmanager-mcp-server --client claude
```

### 3.3. Local Run

- Clone the repository:

```bash
# Clone the repository
$ git clone https://github.com/ntk148v/alertmanager-mcp-server.git
```

- Configure the environment variables for your Prometheus server, either through a .env file or system environment variables:

```shell
# Set environment variables (see .env.sample)
ALERTMANAGER_URL=http://your-alertmanager:9093
ALERTMANAGER_USERNAME=your_username  # optional
ALERTMANAGER_PASSWORD=your_password  # optional
```

#### Transport configuration

You can control how the MCP server communicates with clients using the transport options and host/port settings. These can be set either with command-line flags (which take precedence) or with environment variables.

- MCP_TRANSPORT: Transport mode. One of `stdio`, `http`, or `sse`. Default: `stdio`.
- MCP_HOST: Host/interface to bind when running `http` or `sse` transports (used by the embedded uvicorn server). Default: `0.0.0.0`.
- MCP_PORT: Port to listen on when running `http` or `sse` transports. Default: `8000`.

Examples:

Use environment variables to set defaults (CLI flags still override):

```bash
MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8080 python3 -m src.alertmanager_mcp_server.server
```

Or pass flags directly to override env vars:

```bash
python3 -m src.alertmanager_mcp_server.server --transport http --host 127.0.0.1 --port 9000
```

Notes:
- The `stdio` transport communicates over standard input/output and ignores host/port.
- The `http` (streamable HTTP) and `sse` transports are served via an ASGI app (uvicorn) so host/port are respected when using those transports.

- Add the server configuration to your client configuration file. For example, for Claude Desktop:

```json
{
  "mcpServers": {
    "alertmanager": {
      "command": "uv",
      "args": [
        "--directory",
        "<full path to alertmanager-mcp-server directory>",
        "run",
        "src/alertmanager_mcp_server/server.py"
      ],
      "env": {
        "ALERTMANAGER_URL": "http://your-alertmanager:9093s",
        "ALERTMANAGER_USERNAME": "your_username",
        "ALERTMANAGER_PASSWORD": "your_password"
      }
    }
  }
}
```

- Or install it using make command:

```shell
$ make install
```

- Restart Claude Desktop to load new configuration.
- You can now ask Claude to interact with Alertmanager using natual language:
  - "Show me current alerts"
  - "Filter alerts related to CPU issues"
  - "Get details for this alert"
  - "Create a silence for this alert for the next 2 hours"

![](./images/sample1.jpg)

![](./images/sample2.jpg)

### 3.4. Docker Run

- Run it with pre-built image (or you can build it yourself):

```bash
$ docker run -e ALERTMANAGER_URL=http://your-alertmanager:9093 \
    -e ALERTMANAGER_USERNAME=your_username \
    -e ALERTMANAGER_PASSWORD=your_password \
    -p 8000:8000 ghcr.io/ntk148v/alertmanager-mcp-server
```

- Running with Docker in Claude Desktop:

```json
{
  "mcpServers": {
    "alertmanager": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e", "ALERTMANAGER_URL",
        "-e", "ALERTMANAGER_USERNAME",
        "-e", "ALERTMANAGER_PASSWORD",
        "ghcr.io/ntk148v/alertmanager-mcp-server:latest"
      ],
      "env": {
        "ALERTMANAGER_URL": "http://your-alertmanager:9093s",
        "ALERTMANAGER_USERNAME": "your_username",
        "ALERTMANAGER_PASSWORD": "your_password"
      }
    }
  }
}
```

This configuration passes the environment variables from Claude Desktop to the Docker container by using the `-e` flag with just the variable name, and providing the actual values in the `env` object.

## 4. Tools

The MCP server exposes tools for querying and managing Alertmanager, following [its API v2](https://github.com/prometheus/alertmanager/blob/main/api/v2/openapi.yaml):

- **Get status**: `get_status()`
- **List alerts**: `get_alerts(filter, silenced, inhibited, active, count, offset)`
  - **Pagination support**: Returns paginated results to avoid overwhelming LLM context
  - `count`: Number of alerts per page (default: 10, max: 25)
  - `offset`: Number of alerts to skip (default: 0)
  - Returns: `{ "data": [...], "pagination": { "total": N, "offset": M, "count": K, "has_more": bool } }`
- **List silences**: `get_silences(filter, count, offset)`
  - **Pagination support**: Returns paginated results to avoid overwhelming LLM context
  - `count`: Number of silences per page (default: 10, max: 50)
  - `offset`: Number of silences to skip (default: 0)
  - Returns: `{ "data": [...], "pagination": { "total": N, "offset": M, "count": K, "has_more": bool } }`
- **Create silence**: `post_silence(silence_dict)`
- **Delete silence**: `delete_silence(silence_id)`
- **List receivers**: `get_receivers()`
- **List alert groups**: `get_alert_groups(silenced, inhibited, active, count, offset)`
  - **Pagination support**: Returns paginated results to avoid overwhelming LLM context
  - `count`: Number of alert groups per page (default: 3, max: 5)
  - `offset`: Number of alert groups to skip (default: 0)
  - Returns: `{ "data": [...], "pagination": { "total": N, "offset": M, "count": K, "has_more": bool } }`
  - Note: Alert groups have lower limits because they contain all alerts within each group

### Pagination Benefits

When working with environments that have many alerts, silences, or alert groups, the pagination feature helps:
- **Prevent context overflow**: By default, only 10 items are returned per request
- **Efficient browsing**: LLMs can iterate through results using `offset` and `count` parameters
- **Smart limits**: Maximum of 50 items per page prevents excessive context usage
- **Clear navigation**: `has_more` flag indicates when additional pages are available

Example: If you have 100 alerts, the LLM can fetch them in manageable chunks (e.g., 10 at a time) and only load what's needed for analysis.

See [src/alertmanager_mcp_server/server.py](src/alertmanager_mcp_server/server.py) for full API details.

## 5. Development

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies. Install uv following the instructions for your platform.

```bash
# Clone the repository
$ git clone https://github.com/ntk148v/alertmanager-mcp-server.git
$ cd alertmanager-mcp-server
$ make setup
# Run test
$ make test
# Run in development mode
$ mcp dev
$ TRANSPORT_MODE=sse mcp dev

# Install in Claude Desktop
$ make install
```

## 6. License

[Apache 2.0](LICENSE)

---

<div align="center">
    <sub>Made with ❤️ by <a href="https://github.com/ntk148v">@ntk148v</a></sub>
</div>
