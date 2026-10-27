"""Read-only HTTP server for the dashboard.

It answers GET only and binds to loopback by default. The page is static; everything dynamic is
JSON from /api/*, and the page draws it with textContent, never innerHTML, because script output
and model replies are untrusted text. A strict content security policy backs that up, and the Host
header is checked so a web page cannot reach the server through DNS rebinding.
"""

import hashlib
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kagent.dashboard.reader import WorkspaceReader, json_safe

STATIC_DIR = Path(__file__).parent / "static"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}
NO_CACHE = {"Cache-Control": "no-cache"}
LOOPBACK = ("127.0.0.1", "localhost", "[::1]", "::1")


def is_loopback(host: str) -> bool:
    return host in LOOPBACK


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], reader: WorkspaceReader) -> None:
        super().__init__(address, Handler)
        self.reader = reader
        # Off for a non-loopback bind, where the Host header is whatever the caller typed.
        self.check_host = is_loopback(address[0])
        port = self.server_address[1]
        self.allowed_hosts = {f"{name}:{port}" for name in (address[0], *LOOPBACK)}


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, *args: object) -> None:
        pass  # a poll every two seconds would drown the terminal

    # -- plumbing --------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str, headers: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in {**SECURITY_HEADERS, **headers}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: object, status: int = 200, headers: dict | None = None) -> None:
        body = json.dumps(json_safe(payload), default=str).encode()
        self._send(status, body, "application/json", {**NO_CACHE, **(headers or {})})

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def _host_ok(self) -> bool:
        if not self.server.check_host:
            return True
        return self.headers.get("Host", "").lower() in self.server.allowed_hosts

    # -- routes ----------------------------------------------------------------------------

    def do_GET(self) -> None:
        if not self._host_ok():
            self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
            return
        url = urlparse(self.path)
        try:
            self._route(url.path, parse_qs(url.query))
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "no config.json here; run `kagent init`")
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "bad request")
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")
            raise

    def _route(self, path: str, query: dict[str, list[str]]) -> None:
        reader = self.server.reader
        if path in STATIC:
            name, content_type = STATIC[path]
            self._send(200, (STATIC_DIR / name).read_bytes(), content_type, NO_CACHE)
        elif path == "/api/state":
            etag = '"' + hashlib.sha1(reader.fingerprint().encode()).hexdigest()[:16] + '"'
            if self.headers.get("If-None-Match") == etag:
                self._send(304, b"", "application/json", {**NO_CACHE, "ETag": etag})
            else:
                self._json(reader.state(), headers={"ETag": etag})
        elif path == "/api/events":
            self._json(reader.events(int(query.get("after", ["0"])[0])))
        elif path == "/api/calls":
            self._json(reader.calls(int(query.get("after", ["-1"])[0])))
        elif path.startswith("/api/call/"):
            self._found(reader.call(int(path.removeprefix("/api/call/"))))
        elif path.startswith("/api/node/"):
            self._found(reader.node(path.removeprefix("/api/node/")))
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def _found(self, payload: dict | None) -> None:
        if payload is None:
            self._error(HTTPStatus.NOT_FOUND, "not found")
        else:
            self._json(payload)

    def _refuse(self) -> None:
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "this dashboard is read-only")

    # Everything that could change something is refused.
    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse


def make_server(reader: WorkspaceReader, host: str = "127.0.0.1", port: int = 0):
    return DashboardServer((host, port), reader)


def start_in_background(server: DashboardServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
