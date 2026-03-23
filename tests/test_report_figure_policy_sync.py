from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

from maxionbench.reports import plots


def _extract_prompt_section(text: str, *, start_heading: str, end_heading: str) -> str:
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    return text[start:end]


def _extract_backticked_names(section: str) -> list[str]:
    return re.findall(r"- `([A-Za-z0-9_]+)`", section)


def test_plot_policy_constants_match_prompt() -> None:
    assert plots.FONT_SIZE == 16
    assert plots.PANEL_PX == 600
    assert plots.DPI == 100


def test_plot_style_rcparams_pin_font_sizes_and_palette() -> None:
    plots._set_plot_style()

    assert plt.rcParams["font.size"] == 16
    assert plt.rcParams["axes.titlesize"] == 16
    assert plt.rcParams["axes.labelsize"] == 16
    assert plt.rcParams["xtick.labelsize"] == 16
    assert plt.rcParams["ytick.labelsize"] == 16
    assert plt.rcParams["legend.fontsize"] == 16
    assert plt.rcParams["figure.titlesize"] == 16
    assert plt.rcParams["axes.facecolor"] == plots.FIGURE_FACE_COLOR
    assert plt.rcParams["savefig.facecolor"] == plots.FIGURE_FACE_COLOR
    assert plt.rcParams["grid.color"] == plots.GRID_COLOR
    assert plt.rcParams["text.color"] == plots.TEXT_COLOR
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(plots.ENGINE_PALETTE)


def test_figure_specs_match_prompt_required_figure_names() -> None:
    text = Path("prompt.md").read_text(encoding="utf-8")

    milestone_section = _extract_prompt_section(
        text,
        start_heading="### M1 required figures",
        end_heading="### Final paper figure set (must exist before submission)",
    )
    assert "`m8_deferred_note`" in milestone_section
    milestone_names = _extract_backticked_names(milestone_section)
    milestone_required = milestone_names
    assert [spec.name for spec in plots.MILESTONE_SPECS] == milestone_required

    final_section = _extract_prompt_section(
        text,
        start_heading="### Final paper figure set (must exist before submission)",
        end_heading="## 10) Required outputs per benchmark run",
    )
    final_names = _extract_backticked_names(final_section)
    assert [spec.name for spec in plots.FINAL_SPECS] == final_names


def test_s6_deferred_note_metadata_includes_required_sidecar_fields(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "__meta_config_fingerprint": "cfg-1",
                "engine": "mock",
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D1",
                "seed": 42,
            }
        ]
    )
    paths = plots._write_s6_deferred_note(mode="milestones", out_dir=tmp_path, frame=frame)
    assert [path.name for path in paths] == ["m8_deferred_note.md", "m8_deferred_note.meta.json"]

    payload = json.loads((tmp_path / "m8_deferred_note.meta.json").read_text(encoding="utf-8"))
    assert payload["deferred"] is True
    assert payload["engines"] == ["mock"]
    assert payload["scenarios"] == ["s1_ann_frontier"]
    assert payload["dataset_bundles"] == ["D1"]
    assert payload["seeds"] == [42]
    assert payload["run_ids"] == ["run-1"]
    assert payload["config_fingerprints"] == ["cfg-1"]
