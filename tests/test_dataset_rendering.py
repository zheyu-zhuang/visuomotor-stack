import os

from visuomotor.data.mimicgen.oracle import oracle_cache as OracleCache
from visuomotor.environment._dataset_rendering import (
    orchestration as RenderingOrchestration,
)


def test_worker_cpu_sets_bound_pressure_and_reserve_host_capacity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)))

    assignments = RenderingOrchestration._worker_cpu_sets(12)

    assert len(assignments) == 12
    assert all(assignments)
    assert set().union(*map(set, assignments)) == set(range(28))
    assert sum(map(len, assignments)) == 28
    for index, assigned in enumerate(assignments):
        for other in assignments[index + 1 :]:
            assert set(assigned).isdisjoint(other)


def test_oracle_collector_retains_projection_matrices_for_each_rgb_camera():
    collector = OracleCache.OracleFrameCollector(
        oracle=OracleCache.OracleContext(),
        horizon=3,
        camera_name="agentview",
        camera_names=("agentview", "robot0_eye_in_hand"),
        resolution=256,
        patch_size=16,
        min_patch_area_fraction=0.05,
    )

    arrays = collector.as_arrays()

    assert arrays["camera_matrix_agentview"].shape == (3, 4, 4)
    assert arrays["camera_matrix_robot0_eye_in_hand"].shape == (3, 4, 4)


def test_native_thread_limits_are_scoped_to_worker_spawn(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    for name in RenderingOrchestration._NATIVE_THREAD_ENV[1:]:
        monkeypatch.delenv(name, raising=False)

    with RenderingOrchestration._single_threaded_native_libraries():
        assert all(
            os.environ[name] == "1"
            for name in RenderingOrchestration._NATIVE_THREAD_ENV
        )

    assert os.environ["OMP_NUM_THREADS"] == "8"
    for name in RenderingOrchestration._NATIVE_THREAD_ENV[1:]:
        assert name not in os.environ


def test_worker_setup_pins_cpu_lowers_priority_and_staggers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: calls.append(("affinity", pid, cpus))
    )
    monkeypatch.setattr(os, "nice", lambda value: calls.append(("nice", value)))
    monkeypatch.setattr(
        RenderingOrchestration.cv2,
        "setNumThreads",
        lambda value: calls.append(("cv2", value)),
    )
    monkeypatch.setattr(
        RenderingOrchestration.time,
        "sleep",
        lambda value: calls.append(("sleep", value)),
    )

    RenderingOrchestration._configure_rerender_worker(
        worker_id=3, cpu_ids=(2, 6)
    )

    assert calls == [
        ("affinity", 0, (2, 6)),
        ("nice", 5),
        ("cv2", 1),
        ("sleep", 0.75),
    ]
