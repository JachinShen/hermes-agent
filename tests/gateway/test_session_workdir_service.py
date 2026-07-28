import threading
import time
from types import SimpleNamespace

from tui_gateway.session_workdir import SessionWorkdirContext, SessionWorkdirService


def _service(tmp_path, *, registered=(), persist=None, terminal_apply=None):
    old = tmp_path / "old"; new = tmp_path / "new"; old.mkdir(); new.mkdir()
    sessions = {"s": {"session_key": "s", "cwd": str(old)}}
    def lookup(sid):
        for table_key, session in sessions.items():
            if sid == table_key or sid == session.get("session_key"):
                return session
            if getattr(session.get("agent"), "session_id", None) == sid:
                return session
        return None
    db = {"cwd": str(old)}
    terminal = {"cwd": str(old), "image": "keep"}
    def save(s, cwd):
        if persist: persist()
        db["cwd"] = cwd
    def snap(_): return dict(terminal)
    def apply(_, cwd):
        if terminal_apply: terminal_apply()
        terminal["cwd"] = cwd
    def restore(_, value): terminal.clear(); terminal.update(value)
    ctx = SessionWorkdirContext(
        session_lookup=lookup, registered_paths=lambda: set(registered),
        persist_cwd=save, terminal_snapshot=snap, terminal_apply=apply,
        terminal_restore=restore, git_metadata=lambda *_: None,
        emit_session_info=lambda *_: None, session_info=lambda s: {"cwd": s["cwd"]},
    )
    return SessionWorkdirService(ctx), sessions, db, terminal, old, new


def test_registered_duplicate_path_does_not_mutate_project_state(tmp_path):
    service, sessions, db, terminal, _old, new = _service(tmp_path, registered=[str(tmp_path / "new")])
    projects = {"active_id": "unchanged", "folders": ["same"]}
    receipt = service.switch("s", str(new))
    assert receipt.success
    assert sessions["s"]["cwd"] == str(new)
    assert projects == {"active_id": "unchanged", "folders": ["same"]}


def test_persistence_failure_keeps_live_db_and_terminal(tmp_path):
    def fail(): raise RuntimeError("db failed")
    service, sessions, db, terminal, old, new = _service(tmp_path, registered=[str(tmp_path / "new")], persist=fail)
    receipt = service.switch("s", str(new))
    assert not receipt.success
    assert sessions["s"]["cwd"] == str(old)
    assert db["cwd"] == str(old)
    assert terminal == {"cwd": str(old), "image": "keep"}


def test_terminal_failure_rolls_back_cwd_but_preserves_other_overrides(tmp_path):
    def fail(): raise RuntimeError("terminal failed")
    service, sessions, db, terminal, old, new = _service(tmp_path, registered=[str(tmp_path / "new")], terminal_apply=fail)
    receipt = service.switch("s", str(new))
    assert not receipt.success
    assert sessions["s"]["cwd"] == str(old)
    assert db["cwd"] == str(old)
    assert terminal == {"cwd": str(old), "image": "keep"}


def test_per_session_services_and_locks_do_not_cross_talk(tmp_path):
    service, sessions, _db, _terminal, old, new = _service(tmp_path, registered=[str(tmp_path / "new")])
    sessions["t"] = {"session_key": "t", "cwd": str(old)}
    barrier = threading.Barrier(2)
    original = service.context.persist_cwd
    def save(s, cwd):
        barrier.wait(timeout=2)
        original(s, cwd)
    service.context.persist_cwd = save
    results = []
    threads = [threading.Thread(target=lambda sid: results.append(service.switch(sid, str(new))), args=(sid,)) for sid in ("s", "t")]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert all(item.success for item in results)
    assert sessions["s"]["cwd"] == sessions["t"]["cwd"] == str(new)


def test_relative_path_is_rejected_before_canonicalization(tmp_path):
    service, sessions, db, terminal, old, _new = _service(tmp_path)
    receipt = service.switch("s", "relative/worktree")
    assert not receipt.success
    assert "absolute" in receipt.error
    assert sessions["s"]["cwd"] == str(old)
    assert db["cwd"] == str(old)
    assert terminal["cwd"] == str(old)


def test_missing_session_cwd_and_terminal_failure_leave_no_partial_state(tmp_path):
    def fail():
        raise RuntimeError("terminal failed")

    service, sessions, db, terminal, _old, new = _service(
        tmp_path, registered=[str(tmp_path / "new")], terminal_apply=fail
    )
    sessions["s"].pop("cwd")
    db["cwd"] = None
    receipt = service.switch("s", str(new))
    assert not receipt.success
    assert sessions["s"].get("cwd") is None
    assert db["cwd"] is None
    assert terminal == {"cwd": str(tmp_path / "old"), "image": "keep"}


def test_aliases_for_one_real_session_share_persist_critical_section(tmp_path):
    service, sessions, _db, _terminal, old, new = _service(
        tmp_path, registered=[str(tmp_path / "new")]
    )
    sessions["table-key"] = sessions.pop("s")
    sessions["table-key"]["session_key"] = "durable-key"
    sessions["table-key"]["agent"] = SimpleNamespace(session_id="agent-key")
    active = 0
    max_active = 0
    guard = threading.Lock()
    original = service.context.persist_cwd

    def persist(session, cwd):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            original(session, cwd)
        finally:
            with guard:
                active -= 1

    service.context.persist_cwd = persist
    results = []
    threads = [
        threading.Thread(target=lambda alias: results.append(service.switch(alias, str(new))), args=(alias,))
        for alias in ("table-key", "agent-key")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(item.success for item in results)
    assert max_active == 1
    assert sessions["table-key"]["cwd"] == str(new)
