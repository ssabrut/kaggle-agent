"""HTTP server for the dashboard.

Reading is open to the page: GET only, loopback by default. The page is static; everything dynamic is
JSON from /api/*, and the page draws it with textContent, never innerHTML, because script output
and model replies are untrusted text. A strict content security policy backs that up, and the Host
header is checked so a web page cannot reach the server through DNS rebinding.

Starting and stopping a search are the only actions, and they sit behind four checks: the Host
header, a random token that only this server's own page contains, an Origin equal to the server's
address, and a JSON content type (which makes a browser ask permission first, and we never grant it).
Without a launcher, or on a non-loopback bind, the server is read-only.
"""

import hashlib
import hmac
import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from kagent.dashboard.launch import LaunchError, Launcher, RunRequest
from kagent.dashboard.reader import WorkspaceReader, json_safe

STATIC_DIR = Path(__file__).parent / "static"
STATIC = {
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
POST_ROUTES = ("/api/preflight", "/api/search/start", "/api/search/stop")
MAX_BODY = 16_384
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

    def __init__(
        self, address: tuple[str, int], reader: WorkspaceReader, launcher: Launcher | None = None
    ) -> None:
        if launcher is not None and not is_loopback(address[0]):
            raise ValueError("starting a search from the page is only allowed on a loopback bind")
        super().__init__(address, Handler)
        self.reader = reader
        self.launcher = launcher
        self.token = secrets.token_urlsafe(32)
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

    # -- reading ---------------------------------------------------------------------------

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

    def _index(self) -> bytes:
        """The page, with the token for this server's run controls (or none when read-only)."""
        enabled = self.server.launcher is not None
        page = (STATIC_DIR / "index.html").read_text()
        page = page.replace("__TOKEN__", self.server.token if enabled else "")
        return page.replace("__RUN__", "on" if enabled else "off").encode()

    def _route(self, path: str, query: dict[str, list[str]]) -> None:
        reader = self.server.reader
        if path == "/":
            self._send(200, self._index(), "text/html; charset=utf-8", NO_CACHE)
        elif path in STATIC:
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
        elif path == "/api/launch":
            launcher = self.server.launcher
            self._json(launcher.state() if launcher else {"enabled": False})
        elif path == "/api/runlog":
            self._json(reader.run_log())
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

    # -- acting ----------------------------------------------------------------------------

    def _authorized(self) -> bool:
        token = self.headers.get("X-Kagent-Token", "").encode()
        origin = self.headers.get("Origin", "")
        host = self.headers.get("Host", "")
        right_token = hmac.compare_digest(token, self.server.token.encode())
        return right_token and origin == f"http://{host}"

    def _body(self) -> dict | None:
        """The posted JSON object, or None once an error response has been sent."""
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send application/json")
            return None
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            return None
        if not 0 <= length <= MAX_BODY:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
            return None
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            data = None
        if not isinstance(data, dict):
            self._error(HTTPStatus.BAD_REQUEST, "the body must be a JSON object")
            return None
        return data

    def do_POST(self) -> None:
        if not self._host_ok():
            self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
            return
        path = urlparse(self.path).path
        launcher = self.server.launcher
        if launcher is None or path not in POST_ROUTES:
            self._refuse()
            return
        if not self._authorized():
            self._error(HTTPStatus.FORBIDDEN, "missing or wrong token, or a foreign origin")
            return
        body = self._body()
        if body is None:
            return
        try:
            self._act(path, launcher, body)
        except ValidationError as exc:
            fields = [
                {"field": ".".join(str(part) for part in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ]
            self._json({"error": "invalid settings", "fields": fields}, status=422)
        except LaunchError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))

    def _act(self, path: str, launcher: Launcher, body: dict) -> None:
        if path == "/api/search/stop":
            launcher.stop(force=bool(body.get("force")))
            self._json({"stopping": True})
            return
        request = RunRequest.model_validate(body)
        if path == "/api/preflight":
            self._json(launcher.preflight(request))
        else:
            self._json(launcher.start(request))

    def _refuse(self) -> None:
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "this endpoint does not accept that request")

    # Everything else that could change something is refused.
    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse


def make_server(
    reader: WorkspaceReader,
    host: str = "127.0.0.1",
    port: int = 0,
    launcher: Launcher | None = None,
):
    return DashboardServer((host, port), reader, launcher)


def start_in_background(server: DashboardServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
