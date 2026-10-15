"""The whole inference command on a synthetic volume with random weights: loading, RAS orientation, resampling to
0.5 mm, sliding window, spatial softmax, ostium location, conversion to RAS mm and writing of the output files.
Positions are meaningless (random weights); the test only proves the pipeline runs and writes valid files."""
import json
import pathlib
import subprocess
import sys

from helpers import write_phantom_ct, write_random_checkpoint

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_infer_cli_end_to_end(tmp_path):
    image = tmp_path / "phantom.nii.gz"
    ckpt = tmp_path / "ResNet50_DSNT.pth"
    out = tmp_path / "out"
    write_phantom_ct(image, size=64, spacing_mm=1.0)         # 64 mm cube -> one 128^3 patch at 0.5 mm
    write_random_checkpoint("ResNet50_DSNT", ckpt)

    cmd = [sys.executable, "-m", "hcatostialocalization.infer",
           "-i", str(image), "-m", "ResNet50_DSNT", "-w", str(ckpt),
           "-a", "marker", "markup-3d-slicer", "-o", str(out)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stdout[-3000:] + "\n" + res.stderr[-3000:]

    markers = list(out.rglob("*.ostium.json"))
    slicer = list(out.rglob("*.mrk.json"))
    assert len(markers) == 1 and len(slicer) == 1, (markers, slicer)
    data = json.loads(markers[0].read_text())
    for side in ("L", "R"):
        assert len(data["pos"][side]["ras"]) == 3
        assert len(data["pos"][side]["ijk"]["monai-ras"]) == 3
    assert len(data["affine_from_ijk_to_ras"]["monai-ras"]) == 4
