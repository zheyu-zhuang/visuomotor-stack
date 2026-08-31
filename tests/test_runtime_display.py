import re

from visuomotor.workspace.display import pretty_print_nested

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_runtime_summary_separates_sections_and_aligns_rows(capsys):
    pretty_print_nested(
        {
            "Run": {
                "Task": "square_d0 · in_domain · 400 rollout steps",
                "Stack": "voxel_wrist → voxel_simple → global_flow",
            },
            "Policy": {
                "Generator": "flow matching",
                "Integration": "100 steps",
            },
            "Augmentations": {
                "Scene Yaw": "Enabled ([-180, 180] deg)",
                "Mirror": "Disabled",
            },
            "Training": {
                "Params": "12.34 M encoder",
                "Checkpoint": "experiments/stack_three_d1/checkpoints/latest.ckpt",
            },
        },
        title="Runtime Configuration",
    )

    lines = capsys.readouterr().out.splitlines()
    visible = [_visible(line) for line in lines]

    assert len(set(map(len, visible))) == 1
    assert len(visible[0]) <= 80
    assert visible[1].startswith("│ [Run]")
    assert visible[-2].strip("│ ")

    blanks = [i for i, line in enumerate(visible) if line.strip("│ ") == ""]
    assert len(blanks) == 3
    assert all(visible[i + 1].strip("│ ").startswith("[") for i in blanks)

    assert any("Task : square_d0 · in_domain · 400 rollout steps" in line for line in visible)
    assert any("Generator  : flow matching" in line for line in visible)
    assert any("Scene Yaw: [✓] Enabled ([-180, 180] deg)" in line for line in visible)
    assert any("Mirror   : [×] Disabled" in line for line in visible)
