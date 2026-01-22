######################################################################################################################

"""HTTP transport wrapper for logging with credential redaction."""

######################################################################################################################

import json
import logging
import typing as T

import httpx

######################################################################################################################

# global, re-assignable
L = logging.getLogger("httpx.transport")


def transport_set_logger(logger: logging.Logger) -> None:
    # pylint: disable-next=global-statement
    global L
    if logger is not None:
        L = logger


######################################################################################################################

# need to be lowercase
_REDACTED_KEYS = {
    "emailaddress",
    "password",
    "accesstoken",
    "oldtoken",
    "refreshtoken",
    "idtoken",
    "token",
    "phoneidentifier",
    "userid_type_devicelocalname",
}

_REDACTED_HEADERS = {
    "authorization",
    "cookie",
    "set-cookie",
}


def _redact_obj(obj: T.Any, keys: T.Set[str]) -> T.Any:
    """Recursively redact sensitive fields from objects."""

    if isinstance(obj, httpx.Headers):
        return httpx.Headers([(k, "*********" if k.lower() in keys else v) for k, v in obj.multi_items()])

    if isinstance(obj, dict):
        return {k: "*********" if k.lower() in keys else _redact_obj(v, keys) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_redact_obj(v, keys) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_redact_obj(v, keys) for v in obj)

    return obj


def redact_json_string(raw: str) -> str:
    """Redact sensitive fields from JSON string."""

    try:
        obj = json.loads(raw)
        redacted = _redact_obj(obj, _REDACTED_KEYS)
        return json.dumps(redacted, separators=(",", ":"), indent=2)

    except (json.JSONDecodeError, TypeError):
        return raw[:500]


######################################################################################################################


class HttpxLogTransport(httpx.BaseTransport):
    """Custom HTTP transport that logs requests/responses with credential redaction.

    Wraps the actual transport to intercept requests/responses for debugging.
    Redacts sensitive fields (passwords, tokens, etc.) from logged output.
    """

    def __init__(self, transport: httpx.BaseTransport):
        self.transport = transport

    def log_request(self, req: httpx.Request) -> None:
        L.debug(f"HTTP Request: {req.method} {req.url}")
        L.debug(f"HTTP Request Headers: {_redact_obj(req.headers, _REDACTED_HEADERS)}")

        if req.content:
            try:
                body = req.content.decode("utf-8", errors="replace")
                L.debug(f"HTTP Request Body:\n{redact_json_string(body)}")

            except Exception:  # pylint: disable=broad-except
                L.debug("HTTP Request Body: <binary content>")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle request with logging."""

        if L.isEnabledFor(logging.DEBUG):
            self.log_request(request)

        response = self.transport.handle_request(request)
        response.read()

        if L.isEnabledFor(logging.DEBUG):
            L.debug(f"HTTP Response: {request.url} {response.status_code}")
            L.debug(f"HTTP Response Headers: {_redact_obj(response.headers, _REDACTED_HEADERS)}")

            if response.content:
                try:
                    body = response.content.decode("utf-8", errors="replace")
                    L.debug(f"HTTP Response Body:\n{redact_json_string(body)}")

                except Exception:  # pylint: disable=broad-except
                    L.debug("HTTP Response Body: <binary content>")

        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            extensions=response.extensions,
        )


########################################################################################################################
