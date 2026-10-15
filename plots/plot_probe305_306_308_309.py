#!/usr/bin/env python3
"""Convenient .py entry point for the extended four-probe plot script.

python plots/plot_probe305_306_308_309.py --probe all --no-show
python plots/plot_probe305_306_308_309.py --probe 308 --no-show
python plots/plot_probe305_306_308_309.py --probe 309 --no-show

All options are supplied by the existing plot_probe305_306 script.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().with_name("plot_probe305_306")),run_name="__main__")
