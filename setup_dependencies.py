"""Install Visuomotor Stack's suite dependencies and released assets.

This is a repo-checkout-only tool, like the ``.dep/`` directory it reads
from: it clones and patches the mimicgen-suite repos (robosuite, robomimic,
mimicgen) next to this checkout, and downloads release assets (weights,
textures, backgrounds) from GitHub releases. It intentionally lives outside
the ``visuomotor`` package rather than shipping in the installed wheel,
since none of this is meaningful outside a source checkout.

``visuomotor/cli.py`` loads this module by path (it is not on ``sys.path``)
and drives it via ``vmstack setup``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
PATCHES_DIR = REPO_ROOT / ".dep"
DEFAULT_LOCK_FILE = PATCHES_DIR / "mimicgen.lock"


# ---------------------------------------------------------------------------
# Suite dependency checkouts (robosuite / robomimic / mimicgen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteDependency:
    name: str
    url: str
    revision: str
    patch: str  # "-" means the lock expects no patch on top of `revision`.


def _parse_lock_file(lock_file: Path) -> list:
    dependencies = []
    for line in lock_file.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        name, url, sha, patch = fields
        dependencies.append(SuiteDependency(name=name, url=url, revision=sha, patch=patch))
    return dependencies


def _resolve_patch_path(patch: str, lock_dir: Path) -> Path:
    return Path(patch) if patch.startswith("/") else (lock_dir / patch).resolve()


def _git(path: Path, *args: str) -> str:
    process = subprocess.run(
        ("git", "-C", str(path), *args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _tracked_changes(path: Path) -> str:
    # Untracked files (e.g. robosuite's macros_private.py, created below) are
    # not "local changes" for pin/patch purposes.
    return _git(path, "status", "--porcelain", "--untracked-files=no")


def _patch_already_applied(dst: Path, patch_path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(dst), "apply", "--reverse", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _working_tree_clean(dst: Path) -> bool:
    result = subprocess.run(["git", "-C", str(dst), "diff", "--quiet", "--"])
    return result.returncode == 0


def _ensure_robosuite_macros_private(dst: Path) -> None:
    package_dir = dst / "robosuite"
    macros_file = package_dir / "macros.py"
    private_file = package_dir / "macros_private.py"
    if not macros_file.exists():
        print(
            f"Warning: robosuite macros.py not found at {macros_file}; "
            "skipping macros_private.py setup.",
            file=sys.stderr,
        )
        return
    private_file.write_text(
        "# Created by visuomotor-stack/setup_dependencies.py.\n"
        "# Keep numba cache disabled for editable external checkouts.\n"
        "CACHE_NUMBA = False\n"
    )
    print(f"Configured robosuite macros_private.py: {private_file}")


def _run_quiet(cmd: list, *, error: str) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            f"{error}\n{process.stderr.strip() or process.stdout.strip()}"
        )


def _install_dependency(
    dep: "SuiteDependency", *, stack_root: Path, lock_dir: Path, force: bool = False
) -> None:
    dst = stack_root / dep.name
    if not (dst / ".git").exists():
        print(f"[setup_dependencies] Cloning {dep.name} into {dst}")
        _run_quiet(["git", "clone", dep.url, str(dst)], error=f"Failed to clone {dep.name}")

    current = subprocess.run(
        ["git", "-C", str(dst), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != dep.revision:
        if _tracked_changes(dst):
            if not force:
                raise RuntimeError(
                    f"Refusing to checkout {dep.name}: {dst} has local changes."
                )
        _git(dst, "fetch", "--tags", "origin")
    if force:
        print(f"[setup_dependencies] Restoring {dep.name} to {dep.revision}")
        _git(dst, "checkout", "--force", dep.revision)
    elif current != dep.revision:
        _git(dst, "checkout", dep.revision)

    if dep.patch == "-":
        if _tracked_changes(dst):
            raise RuntimeError(
                f"Refusing to use {dep.name}: {dst} has local changes but the "
                "lock expects no patch."
            )
    else:
        patch_path = _resolve_patch_path(dep.patch, lock_dir)
        if _patch_already_applied(dst, patch_path):
            _git(dst, "apply", "--reverse", str(patch_path))
            clean_after_reverse = _working_tree_clean(dst)
            _git(dst, "apply", str(patch_path))
            if not clean_after_reverse:
                print(
                    f"Patch subset already applied, but {dst} has extra local changes.",
                    file=sys.stderr,
                )
                print(
                    "Reset or recreate this checkout before re-running setup if "
                    "you want the minimal patch set.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            print(f"Patch already applied exactly: {dep.patch}")
        else:
            if _tracked_changes(dst):
                raise RuntimeError(f"Refusing to apply {dep.patch}: {dst} has local changes.")
            _git(dst, "apply", "--check", str(patch_path))
            _git(dst, "apply", str(patch_path))

    if dep.name == "robosuite" and (dst / "robosuite").is_dir():
        _ensure_robosuite_macros_private(dst)

    print(f"[setup_dependencies] Installing {dep.name} (pip install -e)")
    _run_quiet(
        [sys.executable, "-m", "pip", "install", "-e", str(dst)],
        error=f"Failed to pip install {dep.name}",
    )


def _confirm_deps_root(default_path: Path) -> Path:
    answer = input(
        f"[setup_dependencies] Suite dependencies will be installed into "
        f"{default_path}\nPress enter to accept, or type an alternative path: "
    ).strip()
    return Path(answer).expanduser() if answer else default_path


def install_suite_dependencies(
    *,
    deps_root: Optional[str] = None,
    lock_file: Optional[Path] = None,
    force: bool = False,
) -> None:
    """Clone, pin, and patch the mimicgen-suite checkouts, then verify the result."""
    stack_env = os.environ.get("VISUOMOTOR_SUITE_STACK", "mimicgen")
    if stack_env not in ("mimic", "mimicgen"):
        raise RuntimeError(
            f"Unsupported suite stack {stack_env!r}. This branch ships only mimicgen."
        )
    stack_dir_name = "mimic"

    expected_conda_env = os.environ.get("VISUOMOTOR_CONDA_ENV", "vmstack")
    active_conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if expected_conda_env and active_conda_env != expected_conda_env:
        raise RuntimeError(
            f"Expected conda environment '{expected_conda_env}', but active "
            f"environment is '{active_conda_env or 'none'}'. Run 'conda activate "
            f"{expected_conda_env}' first, or set VISUOMOTOR_CONDA_ENV to the target "
            "env name. Set VISUOMOTOR_CONDA_ENV= to skip this check."
        )

    lock_path = lock_file or Path(os.environ.get("VISUOMOTOR_SUITE_LOCK", str(DEFAULT_LOCK_FILE)))
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing suite dependency lock file: {lock_path}")

    deps_root_path = Path(deps_root) if deps_root else REPO_ROOT.parent / "visuomotor-deps"
    if deps_root is None and sys.stdin.isatty():
        deps_root_path = _confirm_deps_root(deps_root_path)
    stack_root = deps_root_path / stack_dir_name
    stack_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Installing {stack_env} suite dependencies from {lock_path} into "
        f"{active_conda_env or 'the active Python environment'}."
    )
    print(f"Checkout directory: {stack_root}")

    for dep in _parse_lock_file(lock_path):
        _install_dependency(
            dep, stack_root=stack_root, lock_dir=lock_path.parent, force=force
        )

    resolved = validate_suite_checkouts(stack_root, lock_file=lock_path)
    print(f"[setup_dependencies] Verified suite checkouts: {resolved}")


def validate_suite_checkouts(root, *, lock_file: Optional[Path] = None) -> dict:
    """Confirm each suite checkout is pinned and patched exactly as the lock expects.

    A checkout that applies a patch is inherently "dirty" from git's
    perspective (the patch changes tracked files without committing), so this
    checks that the tracked diff is *exactly* the pinned patch — via the same
    reverse-apply idempotency check the installer itself uses — rather than
    rejecting any tracked change outright.
    """
    lock_file = lock_file or DEFAULT_LOCK_FILE
    root = Path(root)
    resolved = {}
    for dep in _parse_lock_file(lock_file):
        checkout = root / dep.name
        if not (checkout / ".git").exists():
            raise FileNotFoundError(f"missing dependency checkout: {checkout}")
        head = _git(checkout, "rev-parse", "HEAD")
        if head != dep.revision:
            raise RuntimeError(f"{dep.name} is at {head[:7]}, expected {dep.revision}")
        if dep.patch == "-":
            if _tracked_changes(checkout):
                raise RuntimeError(
                    f"{dep.name} has unexpected local changes (lock expects no patch)"
                )
        else:
            patch_path = _resolve_patch_path(dep.patch, lock_file.parent)
            if not _patch_already_applied(checkout, patch_path):
                raise RuntimeError(f"{dep.name} is missing its pinned patch: {dep.patch}")
        resolved[dep.name] = head
    return resolved


# ---------------------------------------------------------------------------
# Released assets (weights / textures / backgrounds)
# ---------------------------------------------------------------------------

DEFAULT_REPO = "zheyu-zhuang/visuomotor-stack"
DEFAULT_RELEASE_TAG = "assets"
SEEKER_WEIGHTS_ASSET = "seeker.mimicgen.pth"
DINO_WEIGHTS_ASSET = "dinov3.vits16plus.pth"
RVT2_HEATMAP_WEIGHTS_ASSET = "rvt2_heatmap.mimicgen.pth"

# Checksums pin the released weight artifacts (DINOv3 backbone stripped, see
# visuomotor/policy/checkpoint.py::strip_backbone).
EXPECTED_SHA256 = {
    SEEKER_WEIGHTS_ASSET: "7c23050754f35302208832c8fc3c75eab8b34fb48af95e6dc329fe8768dc2ecd",
    DINO_WEIGHTS_ASSET: "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
    RVT2_HEATMAP_WEIGHTS_ASSET: "996ea845bcbb1fc4b5a9e8c66671d83acafcbb89113f74009bf81bc94696212e",
}


def top_level_roots(zf: zipfile.ZipFile) -> set:
    roots = set()
    for name in zf.namelist():
        clean = name.strip("/")
        if clean:
            roots.add(clean.split("/", 1)[0])
    return roots


def extract_archive(archive: Path, target_dir: Path) -> None:
    target_name = target_dir.name

    with zipfile.ZipFile(archive, "r") as zf:
        roots = top_level_roots(zf)
        if target_name in roots:
            extract_root = target_dir.parent
        else:
            extract_root = target_dir

        extract_root.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_root)

    target_dir.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"[setup_assets] Skip existing: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    print(f"[setup_assets] Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"[setup_assets] Saved: {dst}")


def download_release_asset(
    *,
    repo: str,
    release_tag: str,
    asset_name: str,
    dst: Path,
    force: bool,
) -> None:
    if dst.exists() and not force:
        print(f"[setup_assets] Skip existing: {dst}")
        return

    url = f"https://github.com/{repo}/releases/download/{release_tag}/{asset_name}"
    try:
        download_file(url, dst, force=True)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                "Release download returned 404. Check --repo and --release-tag, and "
                "that the release publishes this asset. Missing asset: "
                f"'{asset_name}'."
            ) from exc
        raise


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, asset_name: str) -> None:
    expected = EXPECTED_SHA256.get(asset_name)
    if expected is None:
        return

    actual = sha256_of(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for '{asset_name}': expected {expected}, got "
            f"{actual}. The downloaded file was deleted; re-run vmstack setup --assets-only "
            "to retry, or check --repo/--release-tag if you expect a "
            "different checkpoint."
        )
    print(f"[setup_assets] Checksum OK: {asset_name}")


def warm_task_embedding_cache() -> None:
    """Build the task embedding cache before multi-worker jobs need it."""

    try:
        from visuomotor.data.mimicgen.tasks import (
            _default_cache_path,
            setup_task_embedding_cache,
        )

        cache_path = _default_cache_path()
        if cache_path.exists():
            print(f"[setup_assets] Task embedding cache exists: {cache_path}")
            setup_task_embedding_cache()
        else:
            print(f"[setup_assets] Building task embedding cache: {cache_path}")
            setup_task_embedding_cache()
            print(f"[setup_assets] Task embedding cache saved: {cache_path}")
    except Exception as exc:
        print(
            "[setup_assets] WARNING: task embedding cache was not built. "
            "It will be built lazily by rerender/eval with a file lock. "
            f"Reason: {exc}",
            file=sys.stderr,
        )


def install_assets(
    *,
    repo: str = DEFAULT_REPO,
    release_tag: str = DEFAULT_RELEASE_TAG,
    force: bool = False,
    build_task_cache: bool = True,
) -> int:
    """Download, verify, and install the configured release assets."""

    from visuomotor.paths import BACKGROUNDS_DIR, TEXTURES_DIR, WEIGHTS_DIR

    release_base = f"https://github.com/{repo}/releases/download/{release_tag}"

    seeker_weights = WEIGHTS_DIR / SEEKER_WEIGHTS_ASSET
    dinov3_weights = WEIGHTS_DIR / DINO_WEIGHTS_ASSET
    rvt2_heatmap_weights = WEIGHTS_DIR / RVT2_HEATMAP_WEIGHTS_ASSET

    print(f"[setup_assets] Repo root       : {REPO_ROOT}")
    print(f"[setup_assets] Backgrounds dir : {BACKGROUNDS_DIR}")
    print(f"[setup_assets] Textures dir    : {TEXTURES_DIR}")
    print(f"[setup_assets] Weights dir     : {WEIGHTS_DIR}")
    print(f"[setup_assets] Release source  : {release_base}")

    try:
        with tempfile.TemporaryDirectory(prefix="vmstack_asset_dl_") as temp_dir:
            temp_root = Path(temp_dir)
            backgrounds_zip = temp_root / "backgrounds.zip"
            textures_zip = temp_root / "textures.zip"

            download_release_asset(
                repo=repo,
                release_tag=release_tag,
                asset_name="backgrounds.zip",
                dst=backgrounds_zip,
                force=force,
            )
            download_release_asset(
                repo=repo,
                release_tag=release_tag,
                asset_name="textures.zip",
                dst=textures_zip,
                force=force,
            )

            print("[setup_assets] Extracting backgrounds archive")
            extract_archive(backgrounds_zip, BACKGROUNDS_DIR)
            print("[setup_assets] Extracting textures archive")
            extract_archive(textures_zip, TEXTURES_DIR)

        download_release_asset(
            repo=repo,
            release_tag=release_tag,
            asset_name=SEEKER_WEIGHTS_ASSET,
            dst=seeker_weights,
            force=force,
        )
        verify_checksum(seeker_weights, SEEKER_WEIGHTS_ASSET)
        download_release_asset(
            repo=repo,
            release_tag=release_tag,
            asset_name=DINO_WEIGHTS_ASSET,
            dst=dinov3_weights,
            force=force,
        )
        verify_checksum(dinov3_weights, DINO_WEIGHTS_ASSET)
        download_release_asset(
            repo=repo,
            release_tag=release_tag,
            asset_name=RVT2_HEATMAP_WEIGHTS_ASSET,
            dst=rvt2_heatmap_weights,
            force=force,
        )
        verify_checksum(rvt2_heatmap_weights, RVT2_HEATMAP_WEIGHTS_ASSET)
        if build_task_cache:
            warm_task_embedding_cache()
    except Exception as exc:
        print(f"[setup_assets] Failed: {exc}", file=sys.stderr)
        return 1

    print("[setup_assets] Done.")
    return 0
