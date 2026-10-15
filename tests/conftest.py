"""Shared setup of the smoke tests. They need NO dataset and NO trained weights."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # `import hcatostialocalization` from the repository root
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.environ.setdefault("HCAT_REPO_ROOT", str(ROOT))  # paths of config/machine.yaml
