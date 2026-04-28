from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from reconstructed_minicode.tools.base import ToolDefinition, ToolResult

DANGEROUS_SHELL_CHARS = set('|&;`$(){}<>\n\r')
MAX_MCP_PAYLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

ALLOWED_COMMANDS = {
    'node', 'npm', 'npx', 'python', 'python3', 'pip', 'pip3',
    'uv', 'deno', 'bun', 'cargo', 'go', 'java', 'javac',
    'ruby', 'gem', 'dotnet', 'curl', 'wget',
}

JsonRpcProtocol = str


@dataclass(slots=True)
class McpServerSummary:
    name: str
    command: str
    status: str
    toolCount: int
    error: str | None = None
    protocol: str | None = None
    resourceCount: int | None = None
    promptCount: int | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _sanitize_tool_segment(value: str) -> str:
    normalized = "".join(c.lower() if c.isalnum() or c in {"_", "-"} else "_" for c in value)
    return normalized.strip("_") or "tool"


def _is_absolute_command_allowed(command: str, base_command: str) -> bool:
    """Check if an absolute-path command is in an allowed system directory or whitelist."""
    normalized = Path(command).resolve().as_posix()
    home = Path.home().as_posix()
    allowed_dirs = [
        '/usr/bin', '/usr/local/bin', '/usr/local/sbin', '/usr/sbin', '/opt',
        '/opt/homebrew/bin', '/opt/homebrew/sbin', '/usr/local/Cellar',
        '/snap/bin', '/home/linuxbrew/.linuxbrew/bin',
        f'{home}/.local/bin', f'{home}/.cargo/bin', f'{home}/.nvm',
    ]
    if os.name == 'nt':
        allowed_dirs += ['C:\\Program Files', 'C:\\Program Files (x86)', 'C:\\Windows\\System32']
    return (
        any(normalized.lower().startswith(d.lower()) for d in allowed_dirs)
        or base_command in ALLOWED_COMMANDS
    )


def _validate_mcp_command(command: str) -> None:
    normalized = Path(command).resolve().as_posix()
    if '..' in normalized or '~' in normalized:
        raise RuntimeError(f"Invalid MCP command: contains path traversal characters")

    base_command = Path(command).name.lower()
    for _ext in ('.exe', '.cmd', '.bat'):
        if base_command.endswith(_ext):
            base_command = base_command[:-len(_ext)]
            break

    if Path(command).is_absolute():
        dangerous_shells = ['cmd.exe', 'command.com', 'powershell.exe', 'pwsh.exe']
        if any(normalized.lower().endswith(s) for s in dangerous_shells):
            raise RuntimeError(f'MCP command "{command}" is a dangerous system shell.')
        if not _is_absolute_command_allowed(command, base_command):
            raise RuntimeError(
                f'MCP command "{command}" is not in the allowed list. '
                f'Use a whitelisted command or place the executable in a standard system directory.'
            )
        return

    if base_command not in ALLOWED_COMMANDS:
        raise RuntimeError(
            f'MCP command "{command}" is not in the allowed list. '
            f'Allowed commands: {", ".join(sorted(ALLOWED_COMMANDS))}. '
            f'Use absolute paths for custom commands.'
        )


def _validate_mcp_args(args: list[str]) -> None:
    for arg in args:
        for char in arg:
            if char in DANGEROUS_SHELL_CHARS:
                raise RuntimeError(
                    f"Invalid MCP argument: contains dangerous shell character '{char}'."
                )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _normalize_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    return schema if isinstance(schema, dict) else {"type": "object", "additionalProperties": True}


def _format_content_block(block: Any) -> str:
    if not isinstance(block, dict):
        return json.dumps(block, indent=2, ensure_ascii=False)
    if block.get("type") == "text" and "text" in block:
        return str(block["text"])
    return json.dumps(block, indent=2, ensure_ascii=False)


def _format_tool_call_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=True, output=json.dumps(result, indent=2, ensure_ascii=False))
    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list) and content:
        parts.append("\n\n".join(_format_content_block(b) for b in content))
    if "structuredContent" in result:
        parts.append("STRUCTURED_CONTENT:\n" + json.dumps(result["structuredContent"], indent=2, ensure_ascii=False))
    if not parts:
        parts.append(json.dumps(result, indent=2, ensure_ascii=False))
    return ToolResult(ok=not bool(result.get("isError")), output="\n\n".join(parts).strip())


def _format_read_resource_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=False, output=json.dumps(result, indent=2, ensure_ascii=False))
    contents = result.get("contents", [])
    if not contents:
        return ToolResult(ok=True, output="No resource contents returned.")
    rendered = []
    for item in contents:
        header = f"URI: {item.get('uri', '(unknown)')}"
        if item.get("mimeType"):
            header += f"\nMIME: {item['mimeType']}"
        header += "\n\n"
        if isinstance(item.get("text"), str):
            rendered.append(header + item["text"])
        elif isinstance(item.get("blob"), str):
            rendered.append(header + "BLOB:\n" + item["blob"])
        else:
            rendered.append(header + json.dumps(item, indent=2, ensure_ascii=False))
    return ToolResult(ok=True, output="\n\n".join(rendered))


def _format_prompt_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=False, output=json.dumps(result, indent=2, ensure_ascii=False))
    header = f"DESCRIPTION: {result['description']}\n\n" if result.get("description") else ""
    body_parts = []
    for message in result.get("messages", []):
        role = message.get("role", "unknown")
        content = message.get("content")
        if isinstance(content, str):
            rendered = content
        elif isinstance(content, list):
            rendered = "\n".join(
                str(p["text"]) if isinstance(p, dict) and "text" in p
                else json.dumps(p, indent=2, ensure_ascii=False)
                for p in content
            )
        else:
            rendered = json.dumps(content, indent=2, ensure_ascii=False)
        body_parts.append(f"[{role}]\n{rendered}")
    output = (header + "\n\n".join(body_parts)).strip()
    return ToolResult(ok=True, output=output or json.dumps(result, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------

class StdioMcpClient:
    """MCP client with lazy initialization.

    The server process is not started until the first request is made,
    reducing startup time and resource usage when MCP servers are configured
    but not immediately needed.
    """

    def __init__(self, server_name: str, config: dict[str, Any], cwd: str) -> None:
        self.server_name = server_name
        self.config = config
        self.cwd = cwd
        self.process: subprocess.Popen[bytes] | None = None
        self.protocol: JsonRpcProtocol | None = None
        self.next_id = 1
        self._pending: dict[int, Queue[Any]] = {}
        self._lock = threading.Lock()
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._started = False
        self._start_error: str | None = None
        self._tools_cache: list[dict[str, Any]] | None = None
        self._resources_cache: list[dict[str, Any]] | None = None
        self._prompts_cache: list[dict[str, Any]] | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def start_error(self) -> str | None:
        return self._start_error

    def _protocol_candidates(self) -> list[JsonRpcProtocol]:
        configured = self.config.get("protocol")
        if configured == "content-length":
            return ["content-length"]
        if configured == "newline-json":
            return ["newline-json"]
        return ["content-length", "newline-json"]

    def start(self) -> None:
        """Start the MCP server process (idempotent). Retries after a previous failure."""
        if self._started:
            return
        if self._start_error is not None and self.process is None:
            self._start_error = None

        last_error: Exception | None = None
        for protocol in self._protocol_candidates():
            try:
                self._spawn_process()
                self.protocol = protocol
                self.request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mini-code", "version": "0.1.0"},
                }, timeout_seconds=30.0)
                self.notify("notifications/initialized", {})
                self._started = True
                self._start_error = None
                return
            except Exception as error:  # noqa: BLE001
                last_error = error
                self.close()

        self._start_error = str(last_error or f'Failed to connect MCP server "{self.server_name}".')
        raise RuntimeError(self._start_error)

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()

    def _spawn_process(self) -> None:
        command = str(self.config.get("command", "")).strip()
        if not command:
            raise RuntimeError(f'MCP server "{self.server_name}" has no command configured.')

        # On Windows, bare commands like 'npx'/'npm' are .cmd wrappers — resolve automatically.
        if os.name == "nt" and not Path(command).suffix and not Path(command).is_absolute():
            import shutil
            resolved = shutil.which(command + ".cmd")
            if resolved:
                command = command + ".cmd"

        _validate_mcp_command(command)
        _validate_mcp_args(list(self.config.get("args", []) or []))

        process_cwd = Path(self.cwd)
        if self.config.get("cwd"):
            process_cwd = (process_cwd / str(self.config["cwd"])).resolve()
        env = {**os.environ, **{str(k): str(v) for k, v in (self.config.get("env") or {}).items()}}

        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        try:
            self.process = subprocess.Popen(  # noqa: S603
                [command, *list(self.config.get("args", []) or [])],
                cwd=str(process_cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **popen_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {command}. Install it and ensure it is on PATH.") from None

        self.stderr_lines = []
        with self._lock:
            self._pending = {}
        self._stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
        self._stderr_thread.start()

    def _consume_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            try:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.stderr_lines.append(text)
                    self.stderr_lines = self.stderr_lines[-8:]
            except Exception:
                continue

    def _ensure_stdout_thread(self) -> None:
        if self._stdout_thread is None:
            self._stdout_thread = threading.Thread(target=self._consume_stdout, daemon=True)
            self._stdout_thread.start()

    def _parse_content_length_frame(self, first_line: str) -> dict[str, Any] | None:
        """Read remaining headers + body for one content-length frame."""
        assert self.process is not None and self.process.stdout is not None
        header_lines = [first_line.rstrip("\r\n")]
        while True:
            raw = self.process.stdout.readline()
            if not raw:
                return None
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                return None
            stripped = line.rstrip("\r\n")
            if stripped == "":
                break
            header_lines.append(stripped)

        content_length = 0
        for h in header_lines:
            if h.lower().startswith("content-length:"):
                try:
                    content_length = int(h.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

        if content_length > MAX_MCP_PAYLOAD_BYTES:
            self.stderr_lines.append(f"MCP payload too large: {content_length} bytes (limit {MAX_MCP_PAYLOAD_BYTES})")
            return None
        if content_length <= 0:
            return None

        body = self.process.stdout.read(content_length)
        if len(body) < content_length:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _consume_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue

                if self.protocol is None:
                    self.protocol = "content-length" if line.lower().startswith("content-length:") else "newline-json"

                if self.protocol == "newline-json":
                    try:
                        self._handle_message(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
                else:
                    message = self._parse_content_length_frame(line)
                    if message is not None:
                        self._handle_message(message)
        finally:
            if self.process:
                exit_code = self.process.poll()
                error = {"error": {"code": -1, "message": f"MCP server process exited (code={exit_code})"}}
                with self._lock:
                    for q in list(self._pending.values()):
                        q.put(error)
                    self._pending.clear()

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if not isinstance(message_id, int):
            return
        with self._lock:
            queue = self._pending.pop(message_id, None)
            if queue is not None:
                queue.put(message)

    def send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f'MCP server "{self.server_name}" is not running.')
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        if self.protocol == "newline-json":
            self.process.stdin.write(payload + b"\n")
        else:
            self.process.stdin.write(f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        self.process.stdin.flush()
        self._ensure_stdout_thread()

    def notify(self, method: str, params: Any) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout_seconds: float = 5.0) -> Any:
        message_id = self.next_id
        self.next_id += 1
        queue: Queue[Any] = Queue(maxsize=1)
        with self._lock:
            self._pending[message_id] = queue
        self.send({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params})
        try:
            message = queue.get(timeout=timeout_seconds)
        except Empty as error:
            with self._lock:
                self._pending.pop(message_id, None)
            stderr = "\n".join(self.stderr_lines)
            raise RuntimeError(
                f"MCP {self.server_name}: request timed out for {method}" + (f"\n{stderr}" if stderr else "")
            ) from error
        if message.get("error"):
            details = message["error"].get("data")
            suffix = f"\n{json.dumps(details, indent=2, ensure_ascii=False)}" if details else ""
            raise RuntimeError(f"MCP {self.server_name}: {message['error']['message']}{suffix}")
        return message.get("result")

    def list_tools(self) -> list[dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        self._ensure_started()
        result = self.request("tools/list", {})
        self._tools_cache = list(result.get("tools", []) if isinstance(result, dict) else [])
        return self._tools_cache

    def list_resources(self) -> list[dict[str, Any]]:
        if self._resources_cache is not None:
            return self._resources_cache
        self._ensure_started()
        result = self.request("resources/list", {}, timeout_seconds=3.0)
        self._resources_cache = list(result.get("resources", []) if isinstance(result, dict) else [])
        return self._resources_cache

    def read_resource(self, uri: str) -> ToolResult:
        self._ensure_started()
        return _format_read_resource_result(self.request("resources/read", {"uri": uri}, timeout_seconds=5.0))

    def list_prompts(self) -> list[dict[str, Any]]:
        if self._prompts_cache is not None:
            return self._prompts_cache
        self._ensure_started()
        result = self.request("prompts/list", {}, timeout_seconds=3.0)
        self._prompts_cache = list(result.get("prompts", []) if isinstance(result, dict) else [])
        return self._prompts_cache

    def get_prompt(self, name: str, args: dict[str, str] | None = None) -> ToolResult:
        self._ensure_started()
        return _format_prompt_result(
            self.request("prompts/get", {"name": name, "arguments": args or {}}, timeout_seconds=5.0)
        )

    def call_tool(self, name: str, input_data: Any) -> ToolResult:
        self._ensure_started()
        return _format_tool_call_result(self.request("tools/call", {"name": name, "arguments": input_data or {}}))

    def _terminate_process(self) -> None:
        """Terminate the server process using platform-appropriate strategy."""
        if self.process is None:
            return
        try:
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                        capture_output=True, timeout=5,
                    )
                except Exception:
                    try:
                        self.process.kill()
                    except OSError:
                        pass
            else:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        self.process.kill()
                    except OSError:
                        pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
        finally:
            self.process = None

    def close(self) -> None:
        with self._lock:
            for queue in list(self._pending.values()):
                queue.put({"error": {"message": f'MCP server "{self.server_name}" closed.'}})
            self._pending.clear()

        self._terminate_process()

        self.protocol = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._started = False
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None


# ---------------------------------------------------------------------------
# Tool factory helpers for create_mcp_backed_tools
# ---------------------------------------------------------------------------

def _try_list(fn: Any) -> list[Any]:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return []


def _update_server_status(servers: list[dict[str, Any]], name: str, summary: McpServerSummary) -> None:
    for i, s in enumerate(servers):
        if s["name"] == name:
            servers[i] = asdict(summary)
            return


def _make_mcp_tool(server_name: str, descriptor: dict[str, Any], client: StdioMcpClient) -> ToolDefinition:
    descriptor_name = str(descriptor.get("name", "tool"))
    wrapped_name = f"mcp__{_sanitize_tool_segment(server_name)}__{_sanitize_tool_segment(descriptor_name)}"

    def _run(input_data: Any, _context: Any, *, _c: StdioMcpClient = client, _n: str = descriptor_name) -> ToolResult:
        return _c.call_tool(_n, input_data)

    return ToolDefinition(
        name=wrapped_name,
        description=str(descriptor.get("description") or f"Call MCP tool {descriptor_name} from server {server_name}."),
        input_schema=_normalize_input_schema(descriptor.get("inputSchema")),
        validator=lambda v: v,
        run=_run,
    )


def _make_resource_tools(resource_index: dict[str, dict], clients: list[StdioMcpClient]) -> list[ToolDefinition]:
    if not resource_index:
        return []

    def _list_resources(input_data: dict, _context: Any) -> ToolResult:
        server_filter = input_data.get("server")
        lines = [
            f"{e['serverName']}: {e['resource'].get('uri')}"
            + (f" ({e['resource'].get('name')})" if e["resource"].get("name") else "")
            + (f" - {e['resource'].get('description')}" if e["resource"].get("description") else "")
            for e in resource_index.values()
            if not server_filter or e["serverName"] == server_filter
        ]
        return ToolResult(ok=True, output="\n".join(lines) or "No MCP resources available.")

    def _read_resource(input_data: dict, _context: Any) -> ToolResult:
        client = next((c for c in clients if c.server_name == input_data["server"]), None)
        if client is None:
            return ToolResult(ok=False, output=f"Unknown MCP server: {input_data['server']}")
        return client.read_resource(input_data["uri"])

    return [
        ToolDefinition(
            name="list_mcp_resources",
            description="List available MCP resources exposed by connected MCP servers.",
            input_schema={"type": "object", "properties": {"server": {"type": "string"}}},
            validator=lambda v: {"server": v.get("server")} if isinstance(v, dict) else {"server": None},
            run=_list_resources,
        ),
        ToolDefinition(
            name="read_mcp_resource",
            description="Read a specific MCP resource by server and URI.",
            input_schema={"type": "object", "properties": {"server": {"type": "string"}, "uri": {"type": "string"}}, "required": ["server", "uri"]},
            validator=lambda v: v,
            run=_read_resource,
        ),
    ]


def _make_prompt_tools(prompt_index: dict[str, dict], clients: list[StdioMcpClient]) -> list[ToolDefinition]:
    if not prompt_index:
        return []

    def _list_prompts(input_data: dict, _context: Any) -> ToolResult:
        server_filter = input_data.get("server")
        lines = []
        for e in prompt_index.values():
            if server_filter and e["serverName"] != server_filter:
                continue
            p = e["prompt"]
            args_str = ""
            if p.get("arguments"):
                args_str = " args=[" + ", ".join(
                    f"{a.get('name')}{'*' if a.get('required') else ''}" for a in p["arguments"]
                ) + "]"
            desc_str = f" - {p.get('description')}" if p.get("description") else ""
            lines.append(f"{e['serverName']}: {p.get('name')}{args_str}{desc_str}")
        return ToolResult(ok=True, output="\n".join(lines) or "No MCP prompts available.")

    def _get_prompt(input_data: dict, _context: Any) -> ToolResult:
        client = next((c for c in clients if c.server_name == input_data["server"]), None)
        if client is None:
            return ToolResult(ok=False, output=f"Unknown MCP server: {input_data['server']}")
        return client.get_prompt(input_data["name"], input_data.get("arguments"))

    return [
        ToolDefinition(
            name="list_mcp_prompts",
            description="List available MCP prompts exposed by connected MCP servers.",
            input_schema={"type": "object", "properties": {"server": {"type": "string"}}},
            validator=lambda v: {"server": v.get("server")} if isinstance(v, dict) else {"server": None},
            run=_list_prompts,
        ),
        ToolDefinition(
            name="get_mcp_prompt",
            description="Fetch a rendered MCP prompt by server, prompt name, and optional arguments.",
            input_schema={"type": "object", "properties": {"server": {"type": "string"}, "name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["server", "name"]},
            validator=lambda v: v,
            run=_get_prompt,
        ),
    ]


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _server_config_hash(config: dict) -> str:
    key = f"{config.get('command')}:{':'.join(str(a) for a in config.get('args', []))}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _mcp_cache_path(server_name: str, config: dict) -> Path:
    return Path.home() / ".mini-code" / "mcp-cache" / f"{server_name}-{_server_config_hash(config)}.json"


def _load_disk_cache(server_name: str, config: dict) -> dict[str, list] | None:
    path = _mcp_cache_path(server_name, config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_disk_cache(server_name: str, config: dict, data: dict[str, list]) -> None:
    path = _mcp_cache_path(server_name, config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_mcp_backed_tools(*, cwd: str, mcp_servers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Create MCP-backed tools with lazy server initialization.

    Server processes are not started until the first tool call. A failed server
    does not block other servers and is retried automatically on first use.
    """
    clients: list[StdioMcpClient] = []
    tools: list[ToolDefinition] = []
    servers: list[dict[str, Any]] = []
    resource_index: dict[str, dict[str, Any]] = {}
    prompt_index: dict[str, dict[str, Any]] = {}

    for server_name, config in mcp_servers.items():
        if config.get("enabled") is False:
            servers.append(asdict(McpServerSummary(
                name=server_name, command=config.get("command", ""),
                status="disabled", toolCount=0, protocol=config.get("protocol"),
            )))
            continue

        client = StdioMcpClient(server_name, config, cwd)
        clients.append(client)
        servers.append(asdict(McpServerSummary(
            name=server_name, command=config.get("command", ""),
            status="pending", toolCount=0, protocol=config.get("protocol"),
        )))

        cached = _load_disk_cache(server_name, config)
        if cached is not None:
            descriptors = cached.get("tools", [])
            resources = cached.get("resources", [])
            prompts = cached.get("prompts", [])
            for r in resources:
                resource_index[f"{server_name}:{r.get('uri')}"] = {"serverName": server_name, "resource": r}
            for p in prompts:
                prompt_index[f"{server_name}:{p.get('name')}"] = {"serverName": server_name, "prompt": p}
            tools.extend(_make_mcp_tool(server_name, d, client) for d in descriptors)
            _update_server_status(servers, server_name, McpServerSummary(
                name=server_name, command=config.get("command", ""), status="cached",
                toolCount=len(descriptors), protocol=config.get("protocol"),
                resourceCount=len(resources), promptCount=len(prompts),
            ))
        else:
            try:
                descriptors = client.list_tools()
                resources = _try_list(client.list_resources)
                prompts = _try_list(client.list_prompts)
                _save_disk_cache(server_name, config, {
                    "tools": descriptors, "resources": resources, "prompts": prompts,
                })
                for r in resources:
                    resource_index[f"{server_name}:{r.get('uri')}"] = {"serverName": server_name, "resource": r}
                for p in prompts:
                    prompt_index[f"{server_name}:{p.get('name')}"] = {"serverName": server_name, "prompt": p}
                tools.extend(_make_mcp_tool(server_name, d, client) for d in descriptors)
                _update_server_status(servers, server_name, McpServerSummary(
                    name=server_name, command=config.get("command", ""), status="connected",
                    toolCount=len(descriptors), protocol=client.protocol,
                    resourceCount=len(resources), promptCount=len(prompts),
                ))
            except Exception as error:  # noqa: BLE001
                _update_server_status(servers, server_name, McpServerSummary(
                    name=server_name, command=config.get("command", ""), status="error",
                    toolCount=0, error=str(error)[:200], protocol=config.get("protocol"),
                ))

    tools.extend(_make_resource_tools(resource_index, clients))
    tools.extend(_make_prompt_tools(prompt_index, clients))

    return {
        "tools": tools,
        "servers": servers,
        "dispose": lambda: [c.close() for c in clients],
    }
