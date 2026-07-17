"""Security-hardening tests for execution backends.

Covers the four mitigation domains applied in the hardening commit
(no formatting noise, minimal coverage):

    M1 – fail-closed get_current
    M2 – repo/branch input validation
    M3 – sensitive metadata key expansion & nested filtering
    M4 – CNB runner stdout-only (no stderr echo)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.execution_backends import (
    BackendError,
    BackendRecord,
    BackendStore,
    _default_cnb_runner,
    _is_sensitive_key,
    _sanitize_metadata,
    _validate_create_inputs,
    _validate_git_ref,
    _validate_repo_slug,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "execution_backends.db")


def running_cnb(backend_id: str) -> BackendRecord:
    return BackendRecord(
        id=backend_id,
        driver="cnb",
        status="running",
        repo="example/repo",
        branch="main",
        workspace_sn=f"sn-{backend_id}",
        pipeline_id=f"pipeline-{backend_id}",
        ssh_host="a.example",
        ssh_user="dev",
        ssh_port=22,
        cwd="/workspace",
        created_at="2026-07-17T19:00:00+08:00",
    )


# ── M1: fail-closed get_current ────────────────────────────────────────────


class TestGetCurrentFailClosed:
    """get_current must NOT silently fall back to local when a binding exists."""

    def test_no_binding_returns_local(self, store: BackendStore) -> None:
        """No binding at all → local (unchanged behaviour)."""
        assert store.get_current("unknown-session").id == "local"

    def test_missing_backend_raises(self, store: BackendStore) -> None:
        """Orphaned binding → BackendError (fail-closed)."""
        store._conn.execute(
            "INSERT INTO execution_backend_bindings(session_key, backend_id, updated_at) "
            "VALUES ('session-a', 'nonexistent', '2026-07-18T00:00:00+00:00')"
        )
        with pytest.raises(BackendError, match="unknown backend"):
            store.get_current("session-a")

    def test_not_running_raises(self, store: BackendStore) -> None:
        """Binding to a non-running backend → BackendError (fail-closed)."""
        store.create_backend(running_cnb("cnb-a"))
        store.set_current("session-a", "cnb-a")
        store.update_backend_status("cnb-a", "expired")
        with pytest.raises(BackendError, match="not ready"):
            store.get_current("session-a")

    def test_running_returns_backend(self, store: BackendStore) -> None:
        """Running backend with binding → returned normally (happy path)."""
        store.create_backend(running_cnb("cnb-a"))
        store.set_current("session-a", "cnb-a")
        assert store.get_current("session-a").id == "cnb-a"


# ── M2: repo/branch input validation ───────────────────────────────────────


class TestValidateRepoSlug:
    """_validate_repo_slug rejects dangerous or malformed repo slugs."""

    def test_accepts_simple(self) -> None:
        assert _validate_repo_slug("mygroup/myrepo") == "mygroup/myrepo"

    def test_accepts_subgroup(self) -> None:
        assert _validate_repo_slug("group/sub/repo") == "group/sub/repo"

    def test_accepts_single_segment(self) -> None:
        assert _validate_repo_slug("myrepo") == "myrepo"

    def test_accepts_dots_and_underscores(self) -> None:
        assert _validate_repo_slug("my.team/my_repo") == "my.team/my_repo"

    def test_rejects_empty(self) -> None:
        with pytest.raises(BackendError, match="empty"):
            _validate_repo_slug("")

    def test_rejects_leading_hyphen(self) -> None:
        with pytest.raises(BackendError, match="hyphen"):
            _validate_repo_slug("-group/repo")

    def test_rejects_whitespace(self) -> None:
        with pytest.raises(BackendError, match="slug"):
            _validate_repo_slug("group/my repo")

    def test_rejects_double_slash(self) -> None:
        with pytest.raises(BackendError, match="slug"):
            _validate_repo_slug("group//repo")

    def test_rejects_option_like(self) -> None:
        with pytest.raises(BackendError, match="hyphen"):
            _validate_repo_slug("--help")


class TestValidateGitRef:
    """_validate_git_ref rejects git-check-ref-format violations."""

    def test_accepts_simple(self) -> None:
        assert _validate_git_ref("main") == "main"
        assert _validate_git_ref("feat/foo") == "feat/foo"

    def test_accepts_hyphens_and_dots(self) -> None:
        assert _validate_git_ref("v1.2.3") == "v1.2.3"
        assert _validate_git_ref("my-feature/branch-name") == "my-feature/branch-name"

    def test_rejects_empty(self) -> None:
        with pytest.raises(BackendError, match="empty"):
            _validate_git_ref("")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(BackendError, match="forbidden"):
            _validate_git_ref("foo..bar")

    def test_rejects_tilde(self) -> None:
        with pytest.raises(BackendError, match="forbidden"):
            _validate_git_ref("feature~1")

    def test_rejects_colon(self) -> None:
        with pytest.raises(BackendError, match="forbidden"):
            _validate_git_ref("feature:fix")

    def test_rejects_asterisk(self) -> None:
        with pytest.raises(BackendError, match="forbidden"):
            _validate_git_ref("feature/*/fix")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(BackendError, match="forbidden"):
            _validate_git_ref("feature\\fix")

    def test_rejects_leading_dot(self) -> None:
        with pytest.raises(BackendError, match="dot"):
            _validate_git_ref(".hidden")

    def test_rejects_lock_suffix(self) -> None:
        with pytest.raises(BackendError, match=".lock"):
            _validate_git_ref("main.lock")


class TestValidateCreateInputs:
    """Integration check: _validate_create_inputs calls both validators."""

    def test_valid(self) -> None:
        r, b = _validate_create_inputs("group/repo", "main")
        assert r == "group/repo"
        assert b == "main"

    def test_invalid_repo(self) -> None:
        with pytest.raises(BackendError, match="hyphen"):
            _validate_create_inputs("-group/repo", "main")

    def test_invalid_branch(self) -> None:
        with pytest.raises(BackendError, match="dot"):
            _validate_create_inputs("group/repo", ".hidden")


# ── M3: sensitive metadata key expansion & nested filtering ─────────────────


class TestSensitiveMetadataFiltering:
    """_sanitize_metadata and _is_sensitive_key must block extended key set."""

    SAMPLE_PAYLOAD = {
        "access_url": "https://example.invalid",
        "api_key": "sk-abc123",
        "authorization": "Bearer tok",
        "token": "s3cret",
        "secret_key": "shh",
        "nested": {"inner_token": "leak"},
    }
    FILTERED = frozenset({
        "api_key",
        "authorization",
        "token",
        "secret_key",
        "inner_token",
    })

    def test_original_sensitive_keys_still_filtered(self) -> None:
        sanitized = _sanitize_metadata({
            "authorization": "leak",
            "cookie": "leak",
            "password": "leak",
            "secret": "leak",
            "token": "leak",
        })
        assert sanitized == {}

    def test_new_keys_are_filtered(self) -> None:
        sanitized = _sanitize_metadata({
            "api_key": "leak",
            "apikey": "leak",
            "credential": "leak",
            "credentials": "leak",
            "private_key": "leak",
            "passphrase": "leak",
            "access_key": "leak",
            "secret_key": "leak",
        })
        assert sanitized == {}

    def test_case_insensitive_filtering(self) -> None:
        sanitized = _sanitize_metadata({"API_KEY": "leak", "Token": "leak"})
        assert sanitized == {}

    def test_underscore_hyphen_normalization(self) -> None:
        """Keys with hyphens where underscores are expected must also match."""
        sanitized = _sanitize_metadata({
            "api-key": "leak",
            "secret-key": "leak",
            "access-key": "leak",
        })
        assert sanitized == {}

    def test_nested_dict_filtering(self) -> None:
        """Sensitive keys must be filtered in nested dicts too."""
        sanitized = _sanitize_metadata({
            "visible": "ok",
            "nested": {
                "authorization": "leak",
                "token": "leak",
                "safe": "visible",
            },
        })
        assert sanitized == {
            "visible": "ok",
            "nested": {"safe": "visible"},
        }

    def test_non_sensitive_keys_preserved(self) -> None:
        payload = {"access_url": "ok", "repo": "ok", "branch": "ok"}
        assert _sanitize_metadata(payload) == payload

    def test_list_values_filtered(self) -> None:
        sanitized = _sanitize_metadata([
            {"token": "leak", "safe": "ok"},
            {"api_key": "leak"},
        ])
        assert sanitized == [{"safe": "ok"}, {}]

    def test_non_dict_non_list_unchanged(self) -> None:
        assert _sanitize_metadata("string") == "string"
        assert _sanitize_metadata(42) == 42
        assert _sanitize_metadata(None) is None


# ── M4: CNB runner stdout-only ─────────────────────────────────────────────


class TestDefaultCnbRunnerSecurity:
    """_default_cnb_runner must never echo stderr and must reject empty stdout."""

    def test_normal_stdout_returned(self) -> None:
        result = subprocess.run(
            ["echo", '{"status":200}'],
            capture_output=True, text=True, check=False,
        )
        output = _default_cnb_runner(["echo", '{"status":200}'])
        assert output == '{"status":200}'

    def test_non_zero_exit_no_stderr_leak(self) -> None:
        """Non-zero exit must raise BackendError without stderr content."""
        with pytest.raises(BackendError, match="exited with status"):
            _default_cnb_runner(["bash", "-c", "exit 1"])

    def test_non_zero_exit_with_stderr_no_leak(self) -> None:
        """Stderr content must NOT appear in the error message."""
        with pytest.raises(BackendError, match=r"exited with status 1$"):
            _default_cnb_runner([
                "bash", "-c",
                'echo "secret-token-here" >&2 && exit 1',
            ])

    def test_zero_exit_empty_stdout_raises(self) -> None:
        """Zero exit but empty stdout → fixed error message."""
        with pytest.raises(BackendError, match="no output on stdout"):
            _default_cnb_runner(["bash", "-c", "echo '' && exit 0"])

    def test_stdout_only_ignores_stderr_success(self) -> None:
        """When both stdout and stderr are non-empty, only stdout is used."""
        output = _default_cnb_runner([
            "bash", "-c",
            'echo \'{"status":200}\' && echo "stderr-noise" >&2 && exit 0',
        ])
        assert output == '{"status":200}'


class TestIsSensitiveKey:
    """Unit coverage for _is_sensitive_key helper."""

    def test_exact_match(self) -> None:
        assert _is_sensitive_key("token")

    def test_case_insensitive(self) -> None:
        assert _is_sensitive_key("Token")
        assert _is_sensitive_key("API_KEY")

    def test_hyphen_normalization(self) -> None:
        """Hyphen-delimited keys resolve to the underscore key set."""
        assert _is_sensitive_key("private-key")
        assert _is_sensitive_key("secret-key")

    def test_underscore_normalization(self) -> None:
        assert _is_sensitive_key("private_key")
        assert _is_sensitive_key("secret_key")

    def test_safe_key_not_flagged(self) -> None:
        assert not _is_sensitive_key("safe")
        assert not _is_sensitive_key("access_url")
        assert not _is_sensitive_key("repo")
