# Changelog

## [Unreleased]

### Security

- **Fix unauthenticated access to HTTP/SSE transports** (reported by Syed Anas Mohiuddin, @SyedAnas01). The `http` and `sse` web transports had no MCP-level authentication, bound to `0.0.0.0` by default, and trusted a caller-supplied `X-Scope-OrgId` header to select the Alertmanager tenant. An unauthenticated network caller could silence alerts, delete silences, or inject alerts on any tenant. Changes:
  - **Optional `MCP_API_KEY`**: when set, clients must send `Authorization: Bearer <key>` on every `http`/`sse` request, including SSE `/messages/`. Other requests get `401`. The key is compared in constant time. A startup warning is logged when no key is set.
  - **Default bind host changed from `0.0.0.0` to `127.0.0.1`** so the web transports are loopback-only unless the operator explicitly exposes them.
  - **The tenant is no longer taken from a client-supplied `X-Scope-OrgId` header.** `make_request()` uses only the static `ALERTMANAGER_TENANT` configuration, so a client cannot select another tenant.

### Breaking

- Per-request tenant selection through the `X-Scope-OrgId` request header is removed. Set `ALERTMANAGER_TENANT` instead, and run one server per tenant if you need more than one.
- The `http`/`sse` transports now bind to `127.0.0.1` by default. In Docker, set `MCP_HOST=0.0.0.0` (and `MCP_API_KEY`) to reach the server through a published port.
