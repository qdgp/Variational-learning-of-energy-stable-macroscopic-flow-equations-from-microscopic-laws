"""Prepare star training datasets from LAMMPS dumps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from md_data import train_main


if __name__ == '__main__':
    train_main('star')
