"""Command-line helpers shared by the v1, v2 and compare entry points."""
import os
from typing import List


def expand_paths(items: List[str]) -> List[str]:
    """A single .txt/.list/.lst path expands to its line-separated contents
    (skipping blank lines and '#' comments); anything else is taken as a
    literal path list, made absolute. Mirrors infer.__main__._collect_input_paths.
    """
    if len(items) == 1 and os.path.isfile(items[0]) and items[0].lower().endswith((".txt", ".list", ".lst")):
        collected = []
        with open(items[0], "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                collected.append(os.path.abspath(os.path.expanduser(line)))
        return collected
    return [os.path.abspath(os.path.expanduser(it)) for it in items]


def case_id_from_image_path(image_path: str) -> str:
    name = os.path.basename(image_path)
    for suffix in (".img.nii.gz", ".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]
