import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "visuomotor"
DOMAINS = {
    "action_generation",
    "config",
    "data",
    "environment",
    "geometry",
    "perception",
    "policy",
    "visualization",
    "workspace",
}


def test_package_root_and_domains_are_explicit():
    assert {path.name for path in PACKAGE.iterdir() if path.is_dir() and not path.name.startswith("__")} == DOMAINS
    assert {path.name for path in PACKAGE.glob("*.py")} == {
        "__init__.py",
        "cli.py",
        "paths.py",
    }


def test_dataset_rendering_implementation_is_grouped():
    environment = PACKAGE / "environment"
    implementation = environment / "_dataset_rendering"

    assert {path.name for path in implementation.glob("*.py")} == {
        "__init__.py",
        "cache.py",
        "common.py",
        "orchestration.py",
        "renderer.py",
    }
    assert {path.name for path in environment.glob("dataset_rendering*.py")} == {
        "dataset_rendering.py"
    }
    public_tree = ast.parse((environment / "dataset_rendering.py").read_text())
    public_names = {
        alias.asname or alias.name
        for node in public_tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert public_names == {"render_datasets"}


def test_spatial_producer_contracts_are_data_owned():
    expected = {
        "PointCloudProducerSpec": "data/core/spatial.py",
        "VoxelProducerSpec": "data/core/spatial.py",
    }
    found = {}
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and node.name in expected:
                found[node.name] = str(path.relative_to(PACKAGE))
    assert found == expected
    assert not (PACKAGE / "environment" / "robomimic" / "point_cloud.py").exists()
    assert not (PACKAGE / "environment" / "robomimic" / "voxel.py").exists()


def test_encoder_output_has_one_auxiliary_loss_contract():
    types_tree = ast.parse(
        (PACKAGE / "perception" / "common" / "types.py").read_text()
    )
    output = next(
        node
        for node in types_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EncoderOutput"
    )
    fields = {
        node.target.id
        for node in output.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert "auxiliary_losses" in fields
    assert "auxiliary_loss" not in fields

def test_runner_setup_and_async_protocol_have_single_owners():
    setup = PACKAGE / "environment" / "robomimic" / "robomimic_setup.py"
    setup_tree = ast.parse(setup.read_text())
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "RobomimicRunnerRequest"
        for node in setup_tree.body
    )

    async_path = (
        PACKAGE / "environment" / "gym_wrappers" / "async_vector_env.py"
    )
    async_tree = ast.parse(async_path.read_text())
    workers = {
        node.name: node
        for node in async_tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_worker")
    }
    assert set(workers) == {"_worker", "_worker_shared_memory", "_worker_loop"}
    assert sum(
        isinstance(node, ast.While)
        for worker in workers.values()
        for node in ast.walk(worker)
    ) == 1


def test_retired_namespaces_and_repository_artifacts_are_absent():
    for name in ("model", "util", "focus_toolbox", "generative", "scripts"):
        assert not (PACKAGE / name).exists()
    assert {path.name for path in (ROOT / "docs").iterdir()} == {
        "architecture.md",
        "data.md",
        "development.md",
        "workflows.md",
    }
    assert not (ROOT / "dependencies.lock.toml").exists()
    assert not list(ROOT.glob("*.egg-info"))
    assert not (PACKAGE / "config" / "experiment").exists()
    assert not (PACKAGE / "data" / "mimicgen_dataset.py").exists()
    assert not (PACKAGE / "data" / "mimicgen_voxel_dataset.py").exists()
    assert (PACKAGE / "data" / "mimicgen" / "dataset.py").is_file()
    assert not (PACKAGE / "policy" / "frame_aware.py").exists()
    assert not (PACKAGE / "perception" / "common" / "losses.py").exists()
    assert not (ROOT / "seeker").exists()


def test_only_cli_parses_process_arguments():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path == PACKAGE / "cli.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"parse_args", "parse_known_args"}:
                    violations.append(str(path.relative_to(PACKAGE)))
    assert not violations


def test_cli_is_grouped(capsys):
    import visuomotor.cli as cli

    assert cli.main(["data", "--help"]) == 0
    output = capsys.readouterr().out
    assert "generate-observations" in output
    assert "convert-actions" in output
    assert "playback" in output
    assert "merge" in output
    assert cli.main(["rerender-dataset"]) == 2


def test_package_initializers_are_empty():
    for path in PACKAGE.rglob("__init__.py"):
        if "dinov3_core" not in path.parts:
            assert not ast.parse(path.read_text()).body, path


# Domains that receive concrete Python objects and must never discover their own
# configuration. Workspaces are the Hydra entrypoints, so they are exempt.
RUNTIME_DOMAINS = {
    "action_generation",
    "data",
    "environment",
    "geometry",
    "perception",
    "policy",
    "visualization",
}
CONFIGURATION_MARKERS = (
    "OmegaConf",
    "omegaconf",
    "import hydra",
    "hydra.utils",
    "CONFIG_DIR",
    "PATHS_CONFIG",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = relative.parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _internal_imports(path: Path, modules: set[str]) -> set[str]:
    imports = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            candidates = [
                f"{node.module}.{alias.name}" for alias in node.names
            ] + [node.module]
        else:
            continue
        for candidate in candidates:
            matches = [
                module
                for module in modules
                if candidate == module or candidate.startswith(module + ".")
            ]
            if matches:
                imports.add(max(matches, key=len))
    return imports


def test_runtime_domains_do_not_read_configuration():
    violations = []
    for domain in sorted(RUNTIME_DOMAINS):
        for path in (PACKAGE / domain).rglob("*.py"):
            if "dinov3_core" in path.parts:
                continue
            text = path.read_text()
            found = [marker for marker in CONFIGURATION_MARKERS if marker in text]
            if found:
                violations.append((str(path.relative_to(PACKAGE)), found))
    assert not violations


def test_runtime_dependency_direction_excludes_benchmark_and_builder_leaks():
    forbidden = {
        "perception": ("visuomotor.data.mimicgen",),
        "policy": ("visuomotor.config.build", "visuomotor.data.mimicgen"),
        "environment": ("visuomotor.config", "visuomotor.perception"),
    }
    violations = []
    for domain, prefixes in forbidden.items():
        for path in (PACKAGE / domain).rglob("*.py"):
            imports = _internal_imports(path, set(prefixes))
            if imports:
                violations.append((str(path.relative_to(PACKAGE)), sorted(imports)))
    assert not violations


def test_internal_module_import_graph_is_acyclic():
    paths = list(PACKAGE.rglob("*.py"))
    module_paths = {_module_name(path): path for path in paths}
    graph = {
        module: _internal_imports(path, set(module_paths)).difference({module})
        for module, path in module_paths.items()
    }
    indegree = {module: 0 for module in graph}
    for dependencies in graph.values():
        for dependency in dependencies:
            indegree[dependency] += 1
    ready = [module for module, degree in indegree.items() if degree == 0]
    removed = set()
    while ready:
        module = ready.pop()
        removed.add(module)
        for dependency in graph[module]:
            indegree[dependency] -= 1
            if indegree[dependency] == 0:
                ready.append(dependency)
    assert removed == set(graph), sorted(set(graph).difference(removed))


def test_retired_symbols_and_targets_are_absent():
    forbidden = (
        "ActionCodec",
        "ChunkPolicy",
        "TrajectoryGenerator",
        "DiffusionTrajectoryGenerator",
        "ConditionalFlowGenerator",
        "structured_normalization",
        # Retired configuration routing, replaced by the resolver and the
        # observation contract.
        "config.recipes",
        "recipes.instantiate",
        "config/recipe",
        "focus_profile",
        "focus_method",
        "selected_obs_encoder",
        "enable_voxel_obs",
        "enable_point_cloud_obs",
        "PolicyInputSpec",
        "VoxelObservationTrunk",
        "VoxelCnnBackbone58",
        "cnn58",
        "resnet_focus",
        "environment_metadata",
        "TrainFocusPolicyWorkspace",
        "seeker_stem",
        "SeekerEncoder",
        # Retired by the normalization refactor and collapsed into Normalizer.
        "LinearNormalizer",
        "MultiRobotLinearNormalizer",
        "SingleFieldLinearNormalizer",
        "PoseMap",
        "ActionMap",
        "MimicGenVoxelAnchorDataset",
        "mimicgen_voxel_dataset",
        "FrameAwareGenerativePolicy",
        "VoxelObservationGeometry",
        "gaussian_attention_cross_entropy",
        "plot_switch_points",
        "draw_scores_on_grid",
        "denorm_image",
        "save_attention_heads_video",
        "visualize_trajectory",
        "from seeker ",
        "import seeker ",
        "_target_: seeker.",
        "src.visuomotor",
        "FocusRefineND",
        "FocusND",
    )
    for path in PACKAGE.rglob("*"):
        if path.suffix in {".py", ".yaml"}:
            text = path.read_text()
            assert not [name for name in forbidden if name in text], path


def test_seeker_pretraining_policy_has_no_compatibility_catch_all():
    path = PACKAGE / "policy" / "staged_seeker.py"
    tree = ast.parse(path.read_text())
    policy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SeekerTrainingPolicy"
    )
    initializer = next(
        node
        for node in policy.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    loss = next(
        node
        for node in policy.body
        if isinstance(node, ast.FunctionDef) and node.name == "loss"
    )

    assert initializer.args.kwarg is None
    assert "obs_as_global_cond" not in {
        argument.arg for argument in initializer.args.args
    }
    assert "epoch_idx" in {argument.arg for argument in loss.args.args}
    assert "self.epoch" not in path.read_text()


def test_rollout_source_sensor_names_are_adapter_owned():
    source_sensor_names = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_rot",
        "robot0_gripper_qpos",
    )
    adapter = PACKAGE / "data" / "mimicgen" / "observations.py"
    assert all(name in adapter.read_text() for name in source_sensor_names)

    consumers = (
        PACKAGE / "config" / "resolve.py",
        PACKAGE / "config" / "schema.py",
        PACKAGE / "environment" / "runner.py",
        PACKAGE / "visualization" / "rollout_media.py",
        PACKAGE / "environment" / "robomimic" / "robomimic_setup.py",
        PACKAGE / "environment" / "robomimic" / "robomimic_image_wrapper.py",
    )
    for path in consumers:
        assert not [name for name in source_sensor_names if name in path.read_text()], path
