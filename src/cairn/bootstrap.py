"""
Cairn Bootstrap -- First-session project scanner.

On a user's very first session in a project (0 memories), scans the working
directory and extracts project context: language, framework, dependencies,
recent git activity, directory structure, conventions, and CI setup.

Stores findings as ``project_context`` memories so the agent has useful
knowledge from minute one, solving the cold-start problem.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cairn.bootstrap")

# Manifest files we know how to parse, in priority order.
_MANIFEST_MAP: Dict[str, str] = {
    "pyproject.toml": "Python",
    "setup.py": "Python",
    "package.json": "Node.js",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
    "build.gradle": "Java",
    "Gemfile": "Ruby",
    "mix.exs": "Elixir",
    "Package.swift": "Swift",
    "CMakeLists.txt": "C/C++",
    "Makefile": "C/C++",
}

_LINTER_MAP: Dict[str, str] = {
    "ruff.toml": "ruff",
    ".ruff.toml": "ruff",
    "pyproject.toml": None,  # check [tool.ruff] inside
    ".eslintrc": "ESLint",
    ".eslintrc.js": "ESLint",
    ".eslintrc.json": "ESLint",
    "eslint.config.js": "ESLint",
    "eslint.config.mjs": "ESLint",
    "biome.json": "Biome",
    ".prettierrc": "Prettier",
    "prettier.config.js": "Prettier",
    "rustfmt.toml": "rustfmt",
    ".golangci.yml": "golangci-lint",
}

_TEST_FRAMEWORK_MAP: Dict[str, str] = {
    "pytest.ini": "pytest",
    "conftest.py": "pytest",
    "jest.config.js": "Jest",
    "jest.config.ts": "Jest",
    "vitest.config.ts": "Vitest",
    "vitest.config.js": "Vitest",
    ".mocharc.yml": "Mocha",
    "karma.conf.js": "Karma",
}


# ---------------------------------------------------------------------------
# Scanners — each returns a partial context dict
# ---------------------------------------------------------------------------


def _run_git(args: List[str], cwd: str, timeout: int = 5) -> Optional[str]:
    """Run a git command, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _scan_manifest(project_dir: Path) -> Dict[str, Any]:
    """Detect language, framework, and key dependencies from manifest files."""
    ctx: Dict[str, Any] = {}

    for manifest, language in _MANIFEST_MAP.items():
        manifest_path = project_dir / manifest
        if not manifest_path.exists():
            continue

        ctx["language"] = language
        ctx["manifest_file"] = manifest

        try:
            text = manifest_path.read_text(errors="replace")[:8000]
        except OSError:
            continue

        if manifest == "pyproject.toml":
            _parse_pyproject(text, ctx)
        elif manifest == "package.json":
            _parse_package_json(text, ctx)
        elif manifest == "Cargo.toml":
            _parse_cargo_toml(text, ctx)
        elif manifest == "go.mod":
            _parse_go_mod(text, ctx)

        break  # use first match

    return ctx


def _parse_pyproject(text: str, ctx: Dict[str, Any]) -> None:
    """Extract project info from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Fallback: regex-based extraction
            import re

            name_match = re.search(r'name\s*=\s*"([^"]+)"', text)
            if name_match:
                ctx["project_name"] = name_match.group(1)
            version_match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
            if version_match:
                ctx["python_version"] = version_match.group(1)
            return

    try:
        data = tomllib.loads(text)
    except Exception:
        return

    project = data.get("project", {})
    ctx["project_name"] = project.get("name", "")
    ctx["python_version"] = project.get("requires-python", "")

    # Dependencies
    deps = project.get("dependencies", [])
    if isinstance(deps, list):
        # Extract package names (strip version specifiers)
        import re

        ctx["dependencies"] = [re.split(r"[><=!~\[]", d)[0].strip() for d in deps[:20]]

    # Detect framework from deps
    dep_str = " ".join(ctx.get("dependencies", []))
    for framework, pattern in [
        ("FastAPI", "fastapi"),
        ("Django", "django"),
        ("Flask", "flask"),
        ("Starlette", "starlette"),
    ]:
        if pattern in dep_str.lower():
            ctx["framework"] = framework
            break

    # Check for ruff config
    if "tool" in data and "ruff" in data["tool"]:
        ctx.setdefault("linters", []).append("ruff")


def _parse_package_json(text: str, ctx: Dict[str, Any]) -> None:
    """Extract project info from package.json."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return

    ctx["project_name"] = data.get("name", "")

    # Dependencies
    all_deps: List[str] = []
    for key in ("dependencies", "devDependencies"):
        if isinstance(data.get(key), dict):
            all_deps.extend(data[key].keys())
    ctx["dependencies"] = all_deps[:20]

    # Detect framework
    dep_set = set(d.lower() for d in all_deps)
    for framework, pattern in [
        ("Next.js", "next"),
        ("React", "react"),
        ("Vue", "vue"),
        ("Svelte", "svelte"),
        ("Express", "express"),
        ("Hono", "hono"),
        ("Fastify", "fastify"),
        ("Angular", "@angular/core"),
    ]:
        if pattern in dep_set:
            ctx["framework"] = framework
            break

    # Scripts
    scripts = data.get("scripts", {})
    if isinstance(scripts, dict):
        ctx["scripts"] = {k: v for k, v in list(scripts.items())[:8]}


def _parse_cargo_toml(text: str, ctx: Dict[str, Any]) -> None:
    """Extract project info from Cargo.toml."""
    import re

    name_match = re.search(r'name\s*=\s*"([^"]+)"', text)
    if name_match:
        ctx["project_name"] = name_match.group(1)
    edition_match = re.search(r'edition\s*=\s*"([^"]+)"', text)
    if edition_match:
        ctx["rust_edition"] = edition_match.group(1)


def _parse_go_mod(text: str, ctx: Dict[str, Any]) -> None:
    """Extract project info from go.mod."""
    import re

    module_match = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
    if module_match:
        ctx["project_name"] = module_match.group(1)
    go_match = re.search(r"^go\s+(\S+)", text, re.MULTILINE)
    if go_match:
        ctx["go_version"] = go_match.group(1)


def _scan_git(project_dir: Path) -> Dict[str, Any]:
    """Extract recent git history."""
    ctx: Dict[str, Any] = {}
    cwd = str(project_dir)

    # Check if it's a git repo
    if not (project_dir / ".git").exists():
        return ctx

    ctx["is_git"] = True

    # Recent commits (last 15)
    log = _run_git(["log", "--oneline", "-15"], cwd)
    if log:
        commits = [line.strip() for line in log.split("\n") if line.strip()]
        ctx["recent_commits"] = commits
        ctx["commit_count"] = len(commits)

    # Contributors (last 30 days)
    shortlog = _run_git(
        ["shortlog", "-sn", "--no-merges", "--since=30 days ago"],
        cwd,
    )
    if shortlog:
        contributors = []
        for line in shortlog.strip().split("\n"):
            line = line.strip()
            if line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    contributors.append({"commits": int(parts[0].strip()), "name": parts[1].strip()})
        ctx["contributors"] = contributors

    # Recent activity summary (last 7 days)
    recent = _run_git(["log", "--oneline", "--since=7 days ago"], cwd)
    if recent:
        recent_lines = [l for l in recent.split("\n") if l.strip()]
        ctx["commits_this_week"] = len(recent_lines)

    # Current branch
    branch = _run_git(["branch", "--show-current"], cwd)
    if branch:
        ctx["branch"] = branch

    return ctx


def _scan_structure(project_dir: Path) -> Dict[str, Any]:
    """Detect directory structure and key directories."""
    ctx: Dict[str, Any] = {}
    notable_dirs = []

    for name in ("src", "lib", "app", "tests", "test", "docs", "scripts", ".github", ".gitlab"):
        if (project_dir / name).is_dir():
            notable_dirs.append(name)

    ctx["directories"] = notable_dirs

    # Count files by extension (top-level + one level deep, capped)
    ext_counts: Dict[str, int] = {}
    try:
        for p in project_dir.rglob("*"):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(project_dir).parts[:-1]):
                ext = p.suffix.lower()
                if ext and ext not in (".pyc", ".class", ".o", ".so", ".dylib"):
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
            if sum(ext_counts.values()) > 2000:
                break  # cap scanning
    except OSError:
        pass

    if ext_counts:
        top_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ctx["file_extensions"] = {ext: count for ext, count in top_exts}

    return ctx


def _scan_conventions(project_dir: Path) -> Dict[str, Any]:
    """Detect linters, formatters, CI, and conventions."""
    ctx: Dict[str, Any] = {}

    # Linters
    linters: List[str] = []
    for filename, linter in _LINTER_MAP.items():
        if linter and (project_dir / filename).exists():
            if linter not in linters:
                linters.append(linter)
    if linters:
        ctx["linters"] = linters

    # Test framework
    for filename, framework in _TEST_FRAMEWORK_MAP.items():
        if (project_dir / filename).exists():
            ctx["test_framework"] = framework
            break

    # Test file count
    test_count = 0
    for pattern in ("tests/", "test/", "__tests__/"):
        test_dir = project_dir / pattern
        if test_dir.is_dir():
            try:
                test_count = sum(
                    1
                    for p in test_dir.rglob("*")
                    if p.is_file() and p.suffix in (".py", ".js", ".ts", ".rs", ".go")
                )
            except OSError:
                pass
            break
    if test_count:
        ctx["test_file_count"] = test_count

    # CI/CD
    ci_systems: List[str] = []
    if (project_dir / ".github" / "workflows").is_dir():
        ci_systems.append("GitHub Actions")
    if (project_dir / ".gitlab-ci.yml").exists():
        ci_systems.append("GitLab CI")
    if (project_dir / ".circleci").is_dir():
        ci_systems.append("CircleCI")
    if (project_dir / "Dockerfile").exists():
        ctx["has_docker"] = True
    if ci_systems:
        ctx["ci"] = ci_systems

    # CLAUDE.md presence
    if (project_dir / "CLAUDE.md").exists():
        ctx["has_claude_md"] = True

    # Commit style detection (from recent commits)
    return ctx


def _scan_readme(project_dir: Path) -> Dict[str, Any]:
    """Extract project description from README."""
    ctx: Dict[str, Any] = {}

    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme_path = project_dir / name
        if readme_path.exists():
            try:
                text = readme_path.read_text(errors="replace")[:2000]
                # Extract first meaningful paragraph (skip title/badges)
                lines = text.split("\n")
                paragraph_lines: List[str] = []
                in_paragraph = False
                for line in lines:
                    stripped = line.strip()
                    # Skip headings, badges, empty lines at start
                    if not in_paragraph:
                        if stripped and not stripped.startswith("#") and not stripped.startswith("[![") and not stripped.startswith("!["):
                            in_paragraph = True
                            paragraph_lines.append(stripped)
                    else:
                        if not stripped:
                            break  # end of paragraph
                        paragraph_lines.append(stripped)

                if paragraph_lines:
                    description = " ".join(paragraph_lines)[:200]
                    ctx["description"] = description
            except OSError:
                pass
            break

    return ctx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_project(project_dir: str) -> Dict[str, Any]:
    """Scan a project directory and return structured context.

    Returns a dict with keys like 'language', 'framework', 'project_name',
    'recent_commits', 'contributors', 'linters', 'test_framework', etc.
    All keys are optional; missing means not detected.
    """
    path = Path(project_dir).resolve()
    if not path.is_dir():
        return {}

    context: Dict[str, Any] = {"project_dir": str(path)}

    # Run all scanners, merging results
    for scanner in (_scan_manifest, _scan_git, _scan_structure, _scan_conventions, _scan_readme):
        try:
            context.update(scanner(path))
        except Exception as e:
            logger.debug("Bootstrap scanner %s failed: %s", scanner.__name__, e)

    # Derive project name if not set
    if not context.get("project_name"):
        context["project_name"] = path.name

    return context


def format_summary(context: Dict[str, Any]) -> str:
    """Format scanned context into a concise human-readable summary.

    Returns a multi-line string suitable for the session-start welcome message.
    """
    if not context:
        return ""

    lines: List[str] = []

    # Line 1: Project identity
    name = context.get("project_name", "unknown")
    lang = context.get("language", "")
    framework = context.get("framework", "")
    py_ver = context.get("python_version", "")
    go_ver = context.get("go_version", "")
    rust_ed = context.get("rust_edition", "")

    identity_parts = [name]
    if lang:
        ver = py_ver or go_ver or rust_ed
        identity_parts.append(f"{lang} {ver}".strip() if ver else lang)
    if framework:
        identity_parts.append(framework)
    lines.append(f"  Project: {', '.join(identity_parts)}")

    # Line 2: Testing
    test_fw = context.get("test_framework", "")
    test_count = context.get("test_file_count", 0)
    linters = context.get("linters", [])
    tools_parts: List[str] = []
    if test_fw:
        tf = test_fw
        if test_count:
            tf += f" ({test_count} test files)"
        tools_parts.append(tf)
    if linters:
        tools_parts.extend(linters)
    ci = context.get("ci", [])
    if ci:
        tools_parts.extend(ci)
    if tools_parts:
        lines.append(f"  Tools: {', '.join(tools_parts)}")

    # Line 3: Recent activity
    commits_week = context.get("commits_this_week", 0)
    contributors = context.get("contributors", [])
    recent_commits = context.get("recent_commits", [])

    if commits_week or contributors:
        activity_parts: List[str] = []
        if commits_week:
            activity_parts.append(f"{commits_week} commits this week")
        if contributors:
            names = [c["name"] for c in contributors[:3]]
            if len(contributors) > 3:
                activity_parts.append(f"{len(contributors)} contributors")
            elif len(contributors) > 1:
                activity_parts.append(f"team: {', '.join(names)}")
            else:
                activity_parts.append(f"sole contributor: {names[0]}")
        lines.append(f"  Recent: {', '.join(activity_parts)}")
    elif recent_commits:
        # Summarize the theme of recent commits
        lines.append(f"  History: {len(recent_commits)} recent commits")

    # Line 4: Description (if found in README)
    desc = context.get("description", "")
    if desc:
        if len(desc) > 100:
            desc = desc[:97] + "..."
        lines.append(f"  About: {desc}")

    return "\n".join(lines)


def store_bootstrap(
    context: Dict[str, Any],
    project: str = "",
    entity_id: str = "",
) -> int:
    """Store bootstrap findings as project_context memories.

    Returns the number of memories stored.
    """
    from cairn.bridge import store

    stored = 0

    # Skip if context has no meaningful data
    meaningful_keys = {"language", "framework", "dependencies", "recent_commits", "contributors", "description"}
    if not any(k in context for k in meaningful_keys):
        return 0

    # Memory 1: Project overview
    name = context.get("project_name", Path(project).name if project else "unknown")
    lang = context.get("language", "unknown")
    framework = context.get("framework", "")
    desc = context.get("description", "")

    overview_parts = [f"Project: {name}", f"Language: {lang}"]
    if framework:
        overview_parts.append(f"Framework: {framework}")
    if desc:
        overview_parts.append(f"Description: {desc}")

    try:
        store(
            content="\n".join(overview_parts),
            event_type="project_context",
            metadata={"bootstrap": True, "category": "overview"},
            project=project,
            entity_id=entity_id,
        )
        stored += 1
    except Exception as e:
        logger.debug("Failed to store project overview: %s", e)

    # Memory 2: Tech stack
    deps = context.get("dependencies", [])
    linters = context.get("linters", [])
    test_fw = context.get("test_framework", "")
    ci = context.get("ci", [])
    scripts = context.get("scripts", {})

    stack_parts = []
    if deps:
        stack_parts.append(f"Dependencies: {', '.join(deps[:10])}")
    if test_fw:
        count = context.get("test_file_count", 0)
        stack_parts.append(f"Tests: {test_fw}" + (f" ({count} files)" if count else ""))
    if linters:
        stack_parts.append(f"Linters: {', '.join(linters)}")
    if ci:
        stack_parts.append(f"CI: {', '.join(ci)}")
    if scripts:
        stack_parts.append(f"Scripts: {', '.join(scripts.keys())}")

    if stack_parts:
        try:
            store(
                content="\n".join(stack_parts),
                event_type="project_context",
                metadata={"bootstrap": True, "category": "tech_stack"},
                project=project,
                entity_id=entity_id,
            )
            stored += 1
        except Exception as e:
            logger.debug("Failed to store tech stack: %s", e)

    # Memory 3: Recent activity (if git data available)
    recent_commits = context.get("recent_commits", [])
    contributors = context.get("contributors", [])

    if recent_commits or contributors:
        activity_parts = []
        if contributors:
            for c in contributors[:5]:
                activity_parts.append(f"Contributor: {c['name']} ({c['commits']} commits)")
        if recent_commits:
            activity_parts.append("Recent commits:")
            for commit in recent_commits[:10]:
                activity_parts.append(f"  {commit}")

        try:
            store(
                content="\n".join(activity_parts),
                event_type="project_context",
                metadata={"bootstrap": True, "category": "activity"},
                project=project,
                entity_id=entity_id,
            )
            stored += 1
        except Exception as e:
            logger.debug("Failed to store activity: %s", e)

    # Memory 4: Directory structure + conventions
    dirs = context.get("directories", [])
    exts = context.get("file_extensions", {})
    has_claude = context.get("has_claude_md", False)
    has_docker = context.get("has_docker", False)

    structure_parts = []
    if dirs:
        structure_parts.append(f"Key directories: {', '.join(dirs)}")
    if exts:
        ext_str = ", ".join(f"{ext} ({count})" for ext, count in sorted(exts.items(), key=lambda x: x[1], reverse=True))
        structure_parts.append(f"File types: {ext_str}")
    if has_claude:
        structure_parts.append("Has CLAUDE.md (agent instructions)")
    if has_docker:
        structure_parts.append("Has Dockerfile")

    if structure_parts:
        try:
            store(
                content="\n".join(structure_parts),
                event_type="project_context",
                metadata={"bootstrap": True, "category": "structure"},
                project=project,
                entity_id=entity_id,
            )
            stored += 1
        except Exception as e:
            logger.debug("Failed to store structure: %s", e)

    return stored
