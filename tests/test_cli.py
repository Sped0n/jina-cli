"""CLI smoke tests using real endpoints sparingly.

These intentionally keep live API usage small so CI is less likely to hit rate
limits. Most CLI behavior is covered by local unit tests in ``test_local.py``.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from jina_cli import __version__

RUN_LIVE_API_TESTS = os.environ.get("JINA_RUN_LIVE_TESTS") == "1"
HAS_API_KEY = bool(os.environ.get("JINA_API_KEY"))
TEST_ENV = dict(os.environ)


api_key_required = pytest.mark.skipif(
    not (RUN_LIVE_API_TESTS and HAS_API_KEY),
    reason="Set JINA_RUN_LIVE_TESTS=1 and JINA_API_KEY to enable live API smoke tests",
)


def run_jina(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jina_cli.main", *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=stdin,
        env=TEST_ENV,
    )


def skip_if_transient_api_failure(result: subprocess.CompletedProcess) -> None:
    if result.returncode == 2 and (
        "HTTP 503" in result.stderr or "rate limit" in result.stderr.lower()
    ):
        pytest.skip("Transient live API failure")


def assert_ok_or_skip_transient(result: subprocess.CompletedProcess) -> None:
    skip_if_transient_api_failure(result)
    assert result.returncode == 0


class TestSmoke:
    def test_read_url(self):
        r = run_jina("read", "https://example.com")
        assert r.returncode == 0
        assert "Example Domain" in r.stdout

    def test_primer(self):
        r = run_jina("primer")
        assert r.returncode == 0
        assert r.stdout.strip()

    @api_key_required
    def test_search_json(self):
        r = run_jina("search", "jina ai", "-n", "1", "--json")
        assert_ok_or_skip_transient(r)
        assert '"results"' in r.stdout

    @api_key_required
    def test_embed_json(self):
        r = run_jina("embed", "hello world", "--json")
        assert_ok_or_skip_transient(r)
        assert '"embedding"' in r.stdout


class TestHelp:
    def test_layer0_no_args(self):
        r = run_jina()
        assert r.returncode == 0
        assert "jina read" in r.stderr
        assert "jina search" in r.stderr
        assert r.stdout.strip() == ""

    def test_layer1_no_args(self):
        r = run_jina("embed")
        assert r.returncode == 1
        assert "Usage" in r.stderr or "jina embed" in r.stderr

    def test_layer2_help_flag(self):
        r = run_jina("search", "--help")
        assert r.returncode == 0
        assert "--json" in r.stdout or "--json" in r.stderr

    def test_global_timeout_help_flag(self):
        r = run_jina("--help")
        assert r.returncode == 0
        assert "--timeout" in r.stdout or "--timeout" in r.stderr

    def test_typo_suggestion(self):
        r = run_jina("rea")
        assert r.returncode != 0
        assert "read" in r.stderr.lower() or "rerank" in r.stderr.lower()

    def test_grep_removed(self):
        r = run_jina("grep", "test")
        assert r.returncode != 0
        assert "unknown command" in r.stderr.lower()


class TestVersion:
    def test_version_matches_pyproject(self):
        r = run_jina("--version")
        match = re.search(
            r'^version = "([^"]+)"$',
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
                encoding="utf-8"
            ),
            re.MULTILINE,
        )

        assert match is not None
        assert __version__ == match.group(1)
        assert r.returncode == 0
        assert r.stdout.strip() == f"jina, version {__version__}"
        assert r.stderr == ""


class TestErrorHandling:
    def test_invalid_api_key(self):
        env = {**os.environ, "JINA_API_KEY": "invalid-key"}
        result = subprocess.run(
            [sys.executable, "-m", "jina_cli.main", "embed", "test"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 1
        assert "Fix" in result.stderr or "key" in result.stderr.lower()

    def test_missing_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "JINA_API_KEY"}
        result = subprocess.run(
            [sys.executable, "-m", "jina_cli.main", "embed", "test"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 1
        assert "JINA_API_KEY" in result.stderr


class TestExitCodes:
    def test_success_exit_0(self):
        r = run_jina("primer")
        assert r.returncode == 0

    def test_user_error_exit_1(self):
        r = run_jina("embed")
        assert r.returncode == 1
