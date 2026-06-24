import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr  # We use the Reports API for adding panels

# tab10: 10 maximally distinct colors (no similar hue/brightness pairs)
_COLOR_PALETTE = [mcolors.to_hex(c) for c in plt.cm.tab10.colors]


class RunsetsFactory:
    def __init__(self, entity, project):
        self.entity = entity
        self.project = project
        self._run_colors: dict[str, str] = {}

    def _get_or_assign_color(self, run_name: str) -> str:
        if run_name not in self._run_colors:
            color_index = len(self._run_colors) % len(_COLOR_PALETTE)
            self._run_colors[run_name] = _COLOR_PALETTE[color_index]
        return self._run_colors[run_name]

    def filter_by_run_names(self, run_names: list[str]):
        custom_run_colors = {
            name: self._get_or_assign_color(name) for name in run_names
        }
        return wr.Runset(
            entity=self.entity,
            project=self.project,
            name="Runs",
            filters=f'name in ["{"\", \"".join(run_names)}"]',
            custom_run_colors=custom_run_colors,
        )