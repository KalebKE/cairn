"""Behavioral tests for pre_commit_guard's commit-scope / atomicity check.

The guard blocks (exit 2) oversized multi-directory commits and allows small,
focused ones. (Its peer-coordination half imports the removed cairn_platform
module and fails open, so only the scope check is live in this build.)

Uses a real temporary git repo so the two `git diff --cached` calls the guard
makes return genuine staged-file / shortstat output.
"""
import json
import subprocess

import pytest

from cairn.hooks import pre_commit_guard


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


def _staged_repo(tmp_path, files):
    """Create a git repo, write `files` ({relpath: content}), and stage them."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Tester")
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(repo, "add", "-A")
    return repo


def _run_guard(repo, monkeypatch, tmp_path, command="git commit -m 'x'", tool="Bash"):
    """Invoke pre_commit_guard.main() as if Claude Code fired the hook."""
    monkeypatch.setenv("CAIRN_HOME", str(tmp_path / "home"))  # keep error logs off ~/.cairn
    monkeypatch.setenv("TOOL_NAME", tool)
    monkeypatch.setenv("TOOL_INPUT", json.dumps({"command": command}))
    monkeypatch.setenv("PROJECT_DIR", str(repo))
    monkeypatch.setenv("SESSION_ID", "sess-commit")
    return pre_commit_guard.main()


def _many_files(n_dirs, files_per_dir, lines=1):
    body = "\n".join(f"line {i}" for i in range(lines))
    return {
        f"dir{d}/file{f}.txt": body
        for d in range(n_dirs)
        for f in range(files_per_dir)
    }


# --- triggering ---------------------------------------------------------------

def test_non_bash_tool_ignored(tmp_path, monkeypatch):
    repo = _staged_repo(tmp_path, _many_files(3, 5))  # would block if it ran
    assert _run_guard(repo, monkeypatch, tmp_path, tool="Read") is None


def test_non_commit_command_ignored(tmp_path, monkeypatch):
    repo = _staged_repo(tmp_path, _many_files(3, 5))
    assert _run_guard(repo, monkeypatch, tmp_path, command="git status") is None


# --- allow --------------------------------------------------------------------

def test_small_focused_commit_allowed(tmp_path, monkeypatch):
    repo = _staged_repo(tmp_path, {"src/a.py": "x=1\n", "src/b.py": "y=2\n"})
    # No SystemExit == allowed (peer check is dead → returns cleanly).
    assert _run_guard(repo, monkeypatch, tmp_path) is None


def test_many_files_but_few_dirs_allowed(tmp_path, monkeypatch):
    # 12 files but only 2 dirs → dir threshold (>=3) not met → allowed.
    repo = _staged_repo(tmp_path, _many_files(2, 6))
    assert _run_guard(repo, monkeypatch, tmp_path) is None


# --- block --------------------------------------------------------------------

def test_large_multidir_commit_blocked_by_file_count(tmp_path, monkeypatch, capsys):
    repo = _staged_repo(tmp_path, _many_files(3, 4))  # 12 files, 3 dirs
    with pytest.raises(SystemExit) as ei:
        _run_guard(repo, monkeypatch, tmp_path)
    assert ei.value.code == 2
    assert "[COMMIT-SCOPE] BLOCKED" in capsys.readouterr().out


def test_large_multidir_commit_blocked_by_line_count(tmp_path, monkeypatch, capsys):
    # 3 dirs, few files, but >500 changed lines.
    files = {"a/big.py": "\n".join(str(i) for i in range(300)),
             "b/big.py": "\n".join(str(i) for i in range(300)),
             "c/tiny.py": "1\n"}
    repo = _staged_repo(tmp_path, files)
    with pytest.raises(SystemExit) as ei:
        _run_guard(repo, monkeypatch, tmp_path)
    assert ei.value.code == 2
    assert "BLOCKED" in capsys.readouterr().out


# --- overrides ----------------------------------------------------------------

def test_amend_skips_scope_check(tmp_path, monkeypatch):
    repo = _staged_repo(tmp_path, _many_files(3, 4))  # would block
    assert _run_guard(repo, monkeypatch, tmp_path, command="git commit --amend") is None


def test_skip_scope_check_env_override(tmp_path, monkeypatch):
    repo = _staged_repo(tmp_path, _many_files(3, 4))  # would block
    monkeypatch.setenv("CAIRN_SKIP_SCOPE_CHECK", "1")
    assert _run_guard(repo, monkeypatch, tmp_path) is None
