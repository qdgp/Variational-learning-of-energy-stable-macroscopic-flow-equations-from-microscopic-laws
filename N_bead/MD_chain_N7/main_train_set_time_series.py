"""Prepare adjacent-snapshot chain training pairs from LAMMPS dumps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from md_data import time_series_main


if __name__ == '__main__':
    time_series_main()
