from __future__ import annotations

import gzip
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_VOLUME_SUFFIXES = (".nii.gz", ".nrrd", ".mhd", ".npy", ".npz")


def path_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def read_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    path = Path(path)
    suffix = path_suffix(path)
    if suffix == ".nii.gz":
        return _read_nifti(path)
    if suffix == ".nrrd":
        return _read_nrrd(path)
    if suffix == ".mhd":
        return _read_mhd(path)
    if suffix == ".npy":
        return np.load(path).astype(np.float32, copy=False), (1.0, 1.0, 1.0)
    if suffix == ".npz":
        data = np.load(path)
        key = "volume" if "volume" in data else data.files[0]
        spacing = tuple(float(v) for v in data["spacing_zyx"]) if "spacing_zyx" in data else (1.0, 1.0, 1.0)
        return data[key].astype(np.float32, copy=False), spacing
    raise ValueError(f"unsupported volume type: {path}")


def read_volume_metadata(path: str | Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    volume, spacing = read_volume(path)
    return tuple(int(v) for v in volume.shape[:3]), spacing


def parse_coord(value: Any, side: str | None = None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) < 3:
            return None
        return [float(value[0]), float(value[1]), float(value[2])]

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "-"}:
        return None

    maybe_path = Path(text).expanduser()
    if maybe_path.exists():
        return read_coord_file(maybe_path, side=side)
    if _looks_like_path(text):
        return None

    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 3:
        return [float(nums[0]), float(nums[1]), float(nums[2])]
    return None


def _looks_like_path(text: str) -> bool:
    path_markers = ("/", "\\")
    known_suffixes = SUPPORTED_VOLUME_SUFFIXES + (".json", ".txt", ".csv", ".tsv", ".mrk")
    lower = text.lower()
    return any(marker in text for marker in path_markers) or any(lower.endswith(suffix) for suffix in known_suffixes)


def read_coord_file(path: Path, side: str | None = None) -> list[float] | None:
    suffix = path_suffix(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return coord_from_json(data, side=side)
    if suffix == ".npy":
        return parse_coord(np.load(path), side=side)
    if suffix == ".npz":
        data = np.load(path)
        keys = []
        if side:
            keys.extend([side, side.upper(), f"{side}_zyx", f"coord_{side}", f"coords_{side}"])
        keys.extend(["coord", "coords", "point", "zyx", "prediction", "label"])
        for key in keys:
            if key in data:
                return parse_coord(data[key], side=side)
        return parse_coord(data[data.files[0]], side=side)
    return parse_coord(path.read_text(encoding="utf-8"), side=side)


def coord_from_json(data: Any, side: str | None = None) -> list[float] | None:
    if isinstance(data, list):
        direct = parse_coord(data, side=side)
        if direct is not None:
            return direct
        for item in data:
            coord = coord_from_json(item, side=side)
            if coord is not None:
                return coord
        return None
    if not isinstance(data, dict):
        return None

    if side and isinstance(data.get("ostia"), dict):
        coord = _coord_from_ostia_block(data["ostia"], side)
        if coord is not None:
            return coord

    if side and isinstance(data.get("ostia_coordinates"), dict):
        coord = _coord_from_ostia_block(data["ostia_coordinates"], side)
        if coord is not None:
            return coord

    direct = _coord_from_named_axes(data)
    if direct is not None:
        return direct

    if side:
        for key, value in data.items():
            if _text_matches_side(str(key), side):
                coord = coord_from_json(value, side=None)
                if coord is not None:
                    return coord
                coord = parse_coord(value, side=None)
                if coord is not None:
                    return coord

    if side and _json_label_matches_side(data, side):
        for key in ("coordinate_zyx", "position_zyx", "coords_zyx"):
            if key in data:
                return parse_coord(data[key], side=side)
        for key in ("coordinate_ijk", "ijk"):
            if key in data:
                ijk = parse_coord(data[key], side=side)
                if ijk is not None:
                    return [ijk[2], ijk[1], ijk[0]]
        for key in ("position", "coord", "coords"):
            if key in data:
                return parse_coord(data[key], side=side)

    candidate_keys = []
    if side:
        s = side.upper()
        candidate_keys.extend([s, s.lower(), f"{s}_zyx", f"{s.lower()}_zyx"])
    candidate_keys.extend([
        "coord",
        "coords",
        "coordinate_zyx",
        "prediction_zyx",
        "label_zyx",
        "point",
        "position",
    ])
    for key in candidate_keys:
        if key in data:
            return parse_coord(data[key], side=side)

    for value in data.values():
        coord = coord_from_json(value, side=side)
        if coord is not None:
            return coord
    return None


def _coord_from_ostia_block(block: dict[str, Any], side: str) -> list[float] | None:
    side_names = ("R", "RIGHT", "right", "Right") if side.upper() == "R" else ("L", "LEFT", "left", "Left")
    side_data = None
    for key in side_names:
        if key in block:
            side_data = block[key]
            break
    if side_data is None:
        for key, value in block.items():
            if _text_matches_side(str(key), side):
                side_data = value
                break
    if side_data is None:
        return None

    if isinstance(side_data, dict):
        for key in ("coordinate_zyx", "position_zyx", "coords_zyx"):
            if key in side_data:
                return parse_coord(side_data[key], side=None)
        for key in ("coordinate_ijk", "position_ijk", "coords_ijk", "ijk"):
            if key in side_data:
                ijk = parse_coord(side_data[key], side=None)
                if ijk is not None:
                    return [ijk[2], ijk[1], ijk[0]]
        coord = _coord_from_named_axes(side_data)
        if coord is not None:
            return coord

    coord = parse_coord(side_data, side=None)
    if coord is not None:
        return coord
    return coord_from_json(side_data, side=None)


def _coord_from_named_axes(data: dict[str, Any]) -> list[float] | None:
    lowered = {str(k).lower(): v for k, v in data.items()}

    for keys in (("z", "y", "x"), ("slice", "row", "col")):
        if all(k in lowered for k in keys):
            return [float(lowered[keys[0]]), float(lowered[keys[1]]), float(lowered[keys[2]])]

    for keys in (("i", "j", "k"), ("x", "y", "z")):
        if all(k in lowered for k in keys):
            # Most annotation files store named axes as x/y/z or i/j/k.
            # Internally the analysis code uses z/y/x voxel order.
            return [float(lowered[keys[2]]), float(lowered[keys[1]]), float(lowered[keys[0]])]

    candidate_groups = (
        ("positionijk", "ijk", "index", "voxel", "voxelcoord", "voxelcoords"),
        ("positionras", "ras", "world", "worldcoord", "worldcoords"),
        ("position", "coord", "coords", "point"),
    )
    for group in candidate_groups:
        for key in group:
            if key in lowered:
                coord = parse_coord(lowered[key])
                if coord is not None:
                    if "ijk" in key or "voxel" in key or key == "index":
                        return [coord[2], coord[1], coord[0]]
                    return coord
    return None


def _text_matches_side(text: str, side: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    compact = normalized.replace(" ", "")
    if side.upper() == "R":
        tokens = (
            "right",
            "rightostium",
            "rightcoronaryostium",
            "rightcoronary",
            "rca",
            "destro",
            "dx",
            "ostiumr",
            "rostium",
        )
        single_letters = {"r"}
    else:
        tokens = (
            "left",
            "leftostium",
            "leftcoronaryostium",
            "leftcoronary",
            "lca",
            "sinistro",
            "sx",
            "ostiuml",
            "lostium",
        )
        single_letters = {"l"}
    words = set(normalized.split())
    return any(token in compact for token in tokens) or bool(words & single_letters)


def _json_label_matches_side(data: dict[str, Any], side: str) -> bool:
    side = side.upper()
    text = " ".join(
        str(data.get(key, ""))
        for key in ("label", "name", "description", "id")
        if data.get(key) is not None
    ).lower()
    if not text:
        return False
    return _text_matches_side(text, side)


def distance_mm(a: list[float] | None, b: list[float] | None, spacing_zyx: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None or spacing_zyx is None:
        return None
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) * np.asarray(spacing_zyx, dtype=np.float64)
    return float(np.sqrt(np.sum(diff * diff)))


def parse_triplet(row: dict[str, Any], prefix: str) -> list[float] | None:
    keys = [(f"{prefix}_z", f"{prefix}_y", f"{prefix}_x"), (f"{prefix}_0", f"{prefix}_1", f"{prefix}_2")]
    for triplet in keys:
        if all(k in row and str(row[k]).strip() for k in triplet):
            return [float(row[triplet[0]]), float(row[triplet[1]]), float(row[triplet[2]])]
    value = row.get(prefix)
    if isinstance(value, str) and _looks_like_path(value.strip()):
        return None
    return parse_coord(value)


def parse_spacing(row: dict[str, Any], fallback: tuple[float, float, float] | None = None) -> tuple[float, float, float] | None:
    if all(k in row and str(row[k]).strip() for k in ("spacing_z", "spacing_y", "spacing_x")):
        return (float(row["spacing_z"]), float(row["spacing_y"]), float(row["spacing_x"]))
    spacing = parse_coord(row.get("spacing") or row.get("spacing_zyx"))
    if spacing is not None:
        return (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    return fallback


def parse_shape(row: dict[str, Any], fallback: tuple[int, int, int] | None = None) -> tuple[int, int, int] | None:
    if all(k in row and str(row[k]).strip() for k in ("shape_z", "shape_y", "shape_x")):
        return (int(row["shape_z"]), int(row["shape_y"]), int(row["shape_x"]))
    shape = parse_coord(row.get("shape") or row.get("shape_zyx"))
    if shape is not None:
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    return fallback


def normalize_uint8(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    return np.round(scaled * 255.0).astype(np.uint8)


def heatmap_metrics(heatmap: np.ndarray, spacing_zyx: tuple[float, float, float] | None) -> dict[str, float | list[float] | None]:
    h = np.asarray(heatmap, dtype=np.float64)
    if h.ndim > 3:
        h = np.squeeze(h)
    if h.ndim != 3:
        raise ValueError(f"expected 3D heatmap, got shape {h.shape}")

    finite = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    peak_index = np.unravel_index(int(np.argmax(finite)), finite.shape)
    peak_value = float(finite[peak_index])
    median = float(np.median(finite))
    p95 = float(np.percentile(finite, 95))

    positive = finite - float(np.min(finite))
    total = float(np.sum(positive))
    if total <= 0.0:
        return {
            "heatmap_peak_value": peak_value,
            "heatmap_peak_coord_z": float(peak_index[0]),
            "heatmap_peak_coord_y": float(peak_index[1]),
            "heatmap_peak_coord_x": float(peak_index[2]),
            "heatmap_entropy": None,
            "heatmap_entropy_norm": None,
            "heatmap_spread_mm": None,
            "heatmap_peak_prominence_p95": peak_value - p95,
            "heatmap_peak_prominence_median": peak_value - median,
        }

    prob = positive / total
    entropy = float(-np.sum(prob[prob > 0.0] * np.log(prob[prob > 0.0])))
    entropy_norm = entropy / math.log(prob.size) if prob.size > 1 else 0.0

    z = np.arange(h.shape[0], dtype=np.float64)
    y = np.arange(h.shape[1], dtype=np.float64)
    x = np.arange(h.shape[2], dtype=np.float64)
    cz = float(np.sum(prob * z[:, None, None]))
    cy = float(np.sum(prob * y[None, :, None]))
    cx = float(np.sum(prob * x[None, None, :]))
    if spacing_zyx is None:
        spacing_zyx = (1.0, 1.0, 1.0)
    dz2 = ((z - cz) * spacing_zyx[0]) ** 2
    dy2 = ((y - cy) * spacing_zyx[1]) ** 2
    dx2 = ((x - cx) * spacing_zyx[2]) ** 2
    spread2 = (
        np.sum(prob * dz2[:, None, None])
        + np.sum(prob * dy2[None, :, None])
        + np.sum(prob * dx2[None, None, :])
    )

    return {
        "heatmap_peak_value": peak_value,
        "heatmap_peak_coord_z": float(peak_index[0]),
        "heatmap_peak_coord_y": float(peak_index[1]),
        "heatmap_peak_coord_x": float(peak_index[2]),
        "heatmap_entropy": entropy,
        "heatmap_entropy_norm": float(entropy_norm),
        "heatmap_spread_mm": float(math.sqrt(max(spread2, 0.0))),
        "heatmap_peak_prominence_p95": peak_value - p95,
        "heatmap_peak_prominence_median": peak_value - median,
    }


def _read_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        header = f.read(348)
        sizeof_hdr = struct.unpack("<i", header[:4])[0]
        endian = "<"
        if sizeof_hdr != 348:
            sizeof_hdr = struct.unpack(">i", header[:4])[0]
            endian = ">"
        if sizeof_hdr != 348:
            raise ValueError("invalid NIfTI header")

        dims = struct.unpack(endian + "8h", header[40:56])
        shape_xyz = tuple(int(v) for v in dims[1:4])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        pixdim = struct.unpack(endian + "8f", header[76:108])
        vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
        scl_slope = struct.unpack(endian + "f", header[112:116])[0]
        scl_inter = struct.unpack(endian + "f", header[116:120])[0]
        dtype_map = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
            512: np.uint16,
            768: np.uint32,
        }
        if datatype not in dtype_map:
            raise ValueError(f"unsupported NIfTI datatype: {datatype}")
        f.seek(vox_offset)
        n_vox = int(np.prod(shape_xyz))
        raw = f.read(n_vox * np.dtype(dtype_map[datatype]).itemsize)

    arr = np.frombuffer(raw, dtype=np.dtype(dtype_map[datatype]).newbyteorder(endian), count=n_vox)
    arr = arr.reshape(shape_xyz, order="F").astype(np.float32, copy=False)
    if scl_slope and not math.isclose(scl_slope, 0.0):
        arr = arr * float(scl_slope) + float(scl_inter)
    return np.transpose(arr, (2, 1, 0)), (float(pixdim[3]), float(pixdim[2]), float(pixdim[1]))


def _parse_nrrd_header(raw: bytes) -> tuple[dict[str, str], int]:
    marker = b"\n\n"
    sep = raw.find(marker)
    sep_len = 2
    if sep < 0:
        marker = b"\r\n\r\n"
        sep = raw.find(marker)
        sep_len = 4
    if sep < 0:
        raise ValueError("invalid NRRD header")
    text = raw[:sep].decode("utf-8", errors="replace")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields, sep + sep_len


def _read_nrrd(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    raw = path.read_bytes()
    fields, offset = _parse_nrrd_header(raw)
    sizes = tuple(int(v) for v in fields["sizes"].split())
    encoding = fields.get("encoding", "raw").lower()
    type_name = fields.get("type", "float").lower()
    endian = "<" if fields.get("endian", "little").lower() != "big" else ">"
    dtype_map = {
        "uchar": np.uint8,
        "unsigned char": np.uint8,
        "short": np.int16,
        "signed short": np.int16,
        "ushort": np.uint16,
        "unsigned short": np.uint16,
        "int": np.int32,
        "int32": np.int32,
        "uint": np.uint32,
        "uint32": np.uint32,
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "float64": np.float64,
    }
    if type_name not in dtype_map:
        raise ValueError(f"unsupported NRRD type: {type_name}")
    payload = raw[offset:]
    if "gzip" in encoding or encoding == "gz":
        payload = gzip.decompress(payload)
    elif encoding not in ("raw", "txt", "text", "ascii"):
        raise ValueError(f"unsupported NRRD encoding: {encoding}")
    dtype = np.dtype(dtype_map[type_name]).newbyteorder(endian)
    arr = (
        np.fromstring(payload.decode("utf-8", errors="replace"), sep=" ", dtype=dtype)
        if encoding in ("txt", "text", "ascii")
        else np.frombuffer(payload, dtype=dtype, count=int(np.prod(sizes)))
    )
    arr = arr.reshape(sizes, order="F").astype(np.float32, copy=False)
    volume_zyx = np.transpose(arr, (2, 1, 0)) if arr.ndim == 3 else arr
    return volume_zyx, _nrrd_spacing_zyx(fields)


def _nrrd_spacing_zyx(fields: dict[str, str]) -> tuple[float, float, float]:
    directions = fields.get("space directions", "")
    vectors = re.findall(r"\(([^)]*)\)", directions)
    spacing_xyz: list[float] = []
    for vec in vectors[:3]:
        vals = [float(v) for v in vec.split(",") if v.strip().lower() != "none"]
        spacing_xyz.append(float(np.linalg.norm(vals)) if vals else 1.0)
    if len(spacing_xyz) < 3:
        spacing_xyz = [1.0, 1.0, 1.0]
    return (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])


def _read_mhd(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    fields: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().lower()] = value.strip()
    raw_name = fields.get("elementdatafile")
    if not raw_name:
        raise ValueError("MHD ElementDataFile is missing")
    raw_path = path.parent / raw_name
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw payload: {raw_path}")
    sizes_xyz = tuple(int(v) for v in fields["dimsize"].split())
    spacing_xyz = tuple(float(v) for v in fields.get("elementspacing", "1 1 1").split())
    element_type = fields.get("elementtype", "").upper()
    dtype_map = {
        "MET_UCHAR": np.uint8,
        "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16,
        "MET_SHORT": np.int16,
        "MET_UINT": np.uint32,
        "MET_INT": np.int32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    if element_type not in dtype_map:
        raise ValueError(f"unsupported MHD ElementType: {element_type}")
    endian = ">" if fields.get("elementbyteordermsb", "false").lower() == "true" else "<"
    dtype = np.dtype(dtype_map[element_type]).newbyteorder(endian)
    arr = np.fromfile(raw_path, dtype=dtype, count=int(np.prod(sizes_xyz)))
    arr = arr.reshape(sizes_xyz, order="F").astype(np.float32, copy=False)
    return np.transpose(arr, (2, 1, 0)), (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
