import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_setup_dependencies():
    path = Path(__file__).parents[1] / "setup_dependencies.py"
    spec = importlib.util.spec_from_file_location("test_setup_dependencies_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_force_restores_locked_revision_and_reapplies_patch(tmp_path, monkeypatch):
    setup_dependencies = _load_setup_dependencies()
    checkout = tmp_path / "deps" / "example"
    checkout.mkdir(parents=True)
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    source = checkout / "value.txt"
    source.write_text("original\n")
    _git(checkout, "add", "value.txt")
    _git(checkout, "commit", "-m", "original")
    revision = _git(checkout, "rev-parse", "HEAD")

    source.write_text("patched\n")
    patch = tmp_path / "change.patch"
    patch.write_text(_git(checkout, "diff") + "\n")
    _git(checkout, "checkout", "--", "value.txt")
    source.write_text("local change\n")

    dep = setup_dependencies.SuiteDependency(
        name="example", url="unused", revision=revision, patch=patch.name
    )
    monkeypatch.setattr(setup_dependencies, "_run_quiet", lambda *args, **kwargs: None)

    setup_dependencies._install_dependency(
        dep,
        stack_root=tmp_path / "deps",
        lock_dir=tmp_path,
        force=True,
    )

    assert source.read_text() == "patched\n"
    assert setup_dependencies._patch_already_applied(checkout, patch)
