"""Inject a Plotly renderer-setup cell into the documentation notebooks.

Run once after editing notebook content::

    python docs/_inject_plotly_setup.py

Idempotent: only inserts the cell if it isn't already at index 0.
"""
import json
from pathlib import Path

NOTEBOOKS = [
    Path(__file__).parent / "source" / "_examples" / "notebooks" /
    "Data_manipulation_ECG_PPG.ipynb",
    Path(__file__).parent / "source" / "_examples" / "notebooks" /
    "SQI_pipeline.ipynb",
]

SETUP_CELL_TAG = "plotly-renderer-setup"
SETUP_CELL_SOURCE = [
    "# Configure Plotly to render inline figures via CDN so they display\n",
    "# in the rendered Sphinx output (RTD).  Safe to ignore when running\n",
    "# locally in Jupyter.\n",
    "import plotly.io as pio\n",
    "pio.renderers.default = 'notebook_connected'\n",
]


def already_has_setup(nb) -> bool:
    if not nb["cells"]:
        return False
    first = nb["cells"][0]
    tags = first.get("metadata", {}).get("tags", [])
    return SETUP_CELL_TAG in tags


def inject(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if already_has_setup(nb):
        print(f"[skip] {path.name} already has the setup cell")
        return
    setup_cell = {
        "cell_type": "code",
        "metadata": {"tags": [SETUP_CELL_TAG]},
        "source": SETUP_CELL_SOURCE,
        "outputs": [],
        "execution_count": None,
    }
    nb["cells"].insert(0, setup_cell)
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"[ok]   inserted setup cell into {path.name}")


if __name__ == "__main__":
    for nb_path in NOTEBOOKS:
        if not nb_path.exists():
            print(f"[warn] missing {nb_path}")
            continue
        inject(nb_path)
