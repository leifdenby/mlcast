"""Generate a Graphviz SVG diagram of the included LDCast training config.

Run without arguments to regenerate docs/ldcast_config_diagram.svg:

    uv run python docs/generate_ldcast_config_diagram.py

Run with --check to verify the diagram is up to date:

    uv run python docs/generate_ldcast_config_diagram.py --check
"""

import argparse
import sys
from pathlib import Path

import fiddle.graphviz as fgv

from mlcast.config import ldcast_training_experiment

OUT = Path(__file__).parent / "ldcast_config_diagram.svg"


def main() -> None:
    """Generate or verify the LDCast training config diagram."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the diagram is up to date rather than regenerating it.",
    )
    args = parser.parse_args()

    cfg = ldcast_training_experiment.as_buildable()
    g = fgv.render(cfg, max_str_length=40)
    g.format = "svg"
    new_svg = g.pipe().decode()

    if args.check:
        if not OUT.exists() or OUT.read_text() != new_svg:
            print(
                "docs/ldcast_config_diagram.svg is out of date.\n"
                "Run: uv run python docs/generate_ldcast_config_diagram.py"
            )
            sys.exit(1)
        print("docs/ldcast_config_diagram.svg is up to date.")
    else:
        OUT.write_text(new_svg)
        print(f"Written {OUT}")


if __name__ == "__main__":
    main()
