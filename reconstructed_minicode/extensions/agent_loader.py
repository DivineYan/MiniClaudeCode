"""Agent loader — discovers and loads custom subagent definitions from .mini-code/agents/.

Each agent is a Markdown file with YAML frontmatter:

    ---
    name: code-reviewer
    description: Reviews code for quality and security issues
    tools: read_file, grep_files, list_files
    model: haiku
    max_turns: 8
    ---

    You are a senior code reviewer. Analyze the code and provide
    specific, actionable feedback on quality, security, and best practices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from reconstructed_minicode.utils.logging import get_logger

logger = get_logger("agent_loader")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class AgentDefinition:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None  # None = all tools
    model: str | None                # None = inherit from parent
    max_turns: int = 10
    source_file: str = ""


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition) -> None:
        self._agents[agent.name] = agent
        logger.debug("Registered agent: %s", agent.name)

    def find(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def list(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    import re as _re
    yaml_block = match.group(1)
    body = text[match.end()].strip() if match.end() < len(text) else ""
    body = text[match.end():].strip()

    # Simple key: value parser (no PyYAML dependency)
    meta: dict = {}
    for line in yaml_block.splitlines():
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

    return meta, body


def _parse_tools(tools_str: str) -> list[str] | None:
    """Parse comma or space separated tool names."""
    if not tools_str or tools_str.lower() in ("all", "*", ""):
        return None
    return [t.strip() for t in re.split(r"[,\s]+", tools_str) if t.strip()]


def load_agent_file(path: Path) -> AgentDefinition | None:
    """Load a single agent definition from a .md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot read agent file %s: %s", path, e)
        return None

    meta, body = _parse_frontmatter(text)

    name = meta.get("name", "").strip()
    description = meta.get("description", "").strip()

    if not name:
        name = path.stem
    if not description:
        logger.warning("Agent file %s has no description, skipping", path)
        return None

    tools_raw = meta.get("tools", "")
    allowed_tools = _parse_tools(tools_raw)

    model_raw = meta.get("model", "").strip().lower()
    model = None if model_raw in ("", "inherit") else model_raw

    try:
        max_turns = int(meta.get("max_turns", "10"))
    except ValueError:
        max_turns = 10

    system_prompt = body.strip() or f"You are a specialized agent: {description}"

    return AgentDefinition(
        name=name,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        model=model,
        max_turns=max_turns,
        source_file=str(path),
    )


def discover_agents(cwd: str) -> AgentRegistry:
    """Scan project and user-level agent directories and return an AgentRegistry.

    Search order (lower index = higher priority):
      1. <cwd>/.mini-code/agents/    (project-level)
      2. ~/.mini-code/agents/         (user-level)
    """
    registry = AgentRegistry()

    search_dirs = [
        Path(cwd) / ".mini-code" / "agents",
        Path.home() / ".mini-code" / "agents",
    ]

    # Load in reverse priority order so higher-priority entries overwrite
    for agents_dir in reversed(search_dirs):
        if not agents_dir.is_dir():
            continue
        for md_file in sorted(agents_dir.glob("*.md")):
            agent = load_agent_file(md_file)
            if agent:
                registry.register(agent)
                logger.info("Loaded agent '%s' from %s", agent.name, md_file)

    return registry
