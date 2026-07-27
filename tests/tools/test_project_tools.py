from types import SimpleNamespace

import tools.project_tools as project_tools


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_project_create_without_path_does_not_call_workspace_callback(monkeypatch):
    created = SimpleNamespace(id="p1")
    project = SimpleNamespace(id="p1", slug="p1", name="No Path", folders=[])
    calls = []

    import hermes_cli.projects_db as pdb
    monkeypatch.setattr(project_tools, "_workspace_callback", lambda *args: calls.append(args))
    monkeypatch.setattr(pdb, "connect_closing", lambda: _Connection())
    monkeypatch.setattr(pdb, "create_project", lambda conn, **kwargs: created.id)
    monkeypatch.setattr(pdb, "get_project", lambda conn, pid: project)
    monkeypatch.setattr(
        pdb, "set_active",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not mutate active Project")),
    )

    result = project_tools.project_create("No Path")

    assert '"success": true' in result
    assert calls == []


def test_project_create_with_path_calls_workspace_callback(monkeypatch, tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    created = SimpleNamespace(id="p2")
    project = SimpleNamespace(
        id="p2", slug="p2", name="With Path",
        primary_path=str(path), folders=[],
    )
    calls = []

    import hermes_cli.projects_db as pdb
    monkeypatch.setattr(project_tools, "_workspace_callback", lambda *args: calls.append(args) or True)
    monkeypatch.setattr(pdb, "connect_closing", lambda: _Connection())
    monkeypatch.setattr(pdb, "create_project", lambda conn, **kwargs: created.id)
    monkeypatch.setattr(pdb, "get_project", lambda conn, pid: project)
    monkeypatch.setattr(
        pdb, "set_active",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not mutate active Project")),
    )

    result = project_tools.project_create("With Path", path=str(path), task_id="session")

    assert '"success": true' in result
    assert calls == [("session", str(path), "With Path")]


def test_project_create_keeps_project_success_when_workspace_switch_fails(monkeypatch, tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    project = SimpleNamespace(id="p3", slug="p3", name="Partial", primary_path=str(path), folders=[])
    import hermes_cli.projects_db as pdb
    monkeypatch.setattr(project_tools, "_workspace_callback", lambda *args: False)
    monkeypatch.setattr(pdb, "connect_closing", lambda: _Connection())
    monkeypatch.setattr(pdb, "create_project", lambda conn, **kwargs: "p3")
    monkeypatch.setattr(pdb, "get_project", lambda conn, pid: project)
    monkeypatch.setattr(
        pdb, "set_active",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not mutate active Project")),
    )

    import json
    receipt = json.loads(project_tools.project_create("Partial", str(path), "session"))

    assert receipt["success"] is True
    assert receipt["workspace_switched"] is False
    assert receipt["partial"] is True
    assert receipt["id"] == "p3"


def test_project_switch_selects_path_without_mutating_active_project(monkeypatch, tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    project = SimpleNamespace(
        id="p4", slug="p4", name="Switch", primary_path=str(path), folders=[],
    )
    calls = []

    import hermes_cli.projects_db as pdb
    monkeypatch.setattr(project_tools, "_workspace_callback", lambda *args: calls.append(args) or True)
    monkeypatch.setattr(project_tools, "_resolve", lambda conn, selector: project)
    monkeypatch.setattr(pdb, "connect_closing", lambda: _Connection())
    monkeypatch.setattr(
        pdb, "set_active",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not mutate active Project")),
    )

    import json
    receipt = json.loads(project_tools.project_switch("p4", task_id="session"))

    assert receipt["success"] is True
    assert receipt["workspace_switched"] is True
    assert calls == [("session", str(path), "Switch")]
