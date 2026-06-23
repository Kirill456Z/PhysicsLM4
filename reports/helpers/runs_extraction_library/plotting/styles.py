import matplotlib as mpl
import seaborn as sns

# A muted, scientifically-pleasing qualitative palette.
PALETTE = [
    "#4878d0",  # blue
    "#ee854a",  # orange
    "#6acc65",  # green
    "#d65f5f",  # red
    "#956cb4",  # purple
    "#8c613c",  # brown
    "#dc7ec0",  # pink
    "#797979",  # grey
    "#d5bb67",  # yellow
    "#82c6e2",  # light blue
]


def set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        font="DejaVu Sans",
        font_scale=1.1,
        rc={
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 120,
        },
    )
