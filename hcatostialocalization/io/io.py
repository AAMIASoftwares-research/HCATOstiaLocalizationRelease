import os
import json
import numpy
import torch
import SimpleITK as sitk
import monai
from monai.data import MetaTensor
from monai.transforms import LoadImage, Orientation, Compose


def _root_cause(e: BaseException) -> BaseException:
    """MONAI wraps reader errors in 'applying transform ...': walk the chain to the real one."""
    while e.__cause__ is not None:
        e = e.__cause__
    return e


def sitk_affine_ijk_to_ras(image) -> numpy.ndarray:
    """4x4 IJK -> RAS world (mm) affine of a SimpleITK image (or ImageFileReader after ReadImageInformation)."""
    direction = numpy.array(image.GetDirection(), dtype=float).reshape(3, 3)
    affine_lps = numpy.eye(4)
    affine_lps[:3, :3] = direction @ numpy.diag(image.GetSpacing())
    affine_lps[:3, 3] = image.GetOrigin()
    return numpy.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps


_warned_fallback: set = set()

def load_image_ras(image_path, force_sitk: bool = False, verbose: bool = False) -> MetaTensor:
    """
    Loads an image as a MONAI MetaTensor of shape (1, R, A, S) oriented to RAS, with its
    IJK -> RAS affine. Used by the dataset classes and by compute_image_metadata, so that
    all code paths see exactly the same voxel grid.

    1. MONAI LoadImage (reader chosen by extension: nibabel for NIfTI, pynrrd/itk for NRRD, ...).
    2. If MONAI fails, the root cause is printed (once per extension) and the file is read with
       SimpleITK instead: voxel array transposed from (k, j, i) to (i, j, k), affine built from
       the SimpleITK header in RAS, then the same MONAI Orientation("RAS") is applied.

    Why the two paths are voxel-identical: before Orientation, both give array[i, j, k] =
    voxel (i, j, k) as stored in the file (SimpleITK's numpy view is [k, j, i], hence the
    transpose; neither reader flips anything) and the same affine
    diag(-1, -1, 1) @ [Direction @ diag(spacing) | origin]. Orientation("RAS") is a
    deterministic function of (array, affine), so the outputs are identical.
    force_sitk=True skips MONAI, to verify this on a file both readers can open.
    verbose=True prints a notice (once per extension) when MONAI fails and SimpleITK is used instead.
    """
    orientation = Orientation(axcodes="RAS", labels=None)
    try:
        if force_sitk:  # only for checking that both paths give identical results
            raise RuntimeError("force_sitk=True")
        img = LoadImage(image_only=True, ensure_channel_first=True)(image_path)
    except Exception as e:
        ext = os.path.basename(str(image_path)).split(".", 1)[-1].lower()
        if ext not in _warned_fallback and not force_sitk:
            _warned_fallback.add(ext)
            cause = _root_cause(e)
            if verbose:
                print(f"[io] MONAI LoadImage failed on '.{ext}' ({image_path}): {type(cause).__name__}: {cause}\n"
                      f"[io] Falling back to SimpleITK for '.{ext}' files (same RAS result).")
        sitk_img = sitk.ReadImage(str(image_path))
        array = sitk.GetArrayFromImage(sitk_img).transpose(2, 1, 0)  # (k, j, i) -> (i, j, k)
        img = MetaTensor(
            torch.as_tensor(numpy.ascontiguousarray(array), dtype=torch.float32)[None],
            affine=torch.as_tensor(sitk_affine_ijk_to_ras(sitk_img), dtype=torch.float64),
            meta={"filename_or_obj": str(image_path), "space": "RAS"},
        )
        del sitk_img, array
    return orientation(img)




def create_blueprint_dict(image_id, image_filename, pos_dict, affine_ijk_to_ras, affine_ras_to_ijk, comment="", other=None):
    """
    Creates and returns the dictionary formatted according to the blueprint schema.
    """
    if other is None:
        other = {}
        
    blueprint = {
        "version": "1.0",
        "image_id": image_id,
        "image_filename": image_filename,
        "pos": pos_dict,
        "affine_from_ijk_to_ras": affine_ijk_to_ras,
        "affine_from_ras_to_ijk": affine_ras_to_ijk,
        "comment": comment,
        "other": other
    }
    return blueprint



def save_blueprint_json(filepath, image_id, image_filename, pos_dict, affine_ijk_to_ras, affine_ras_to_ijk, comment="", other=None, verbose: bool = False):
    """
    Creates the blueprint dictionary and saves it to a JSON file.
    Includes I/O error handling and checks for directory existence.

    filepath should end (mandatory) in .ostium.json
    verbose=True prints a message when the file has been saved (errors are always printed).
    """
    # Check if parent directory exists, if not, create it
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir)
        except OSError as e:
            print(f"Error creating directory {parent_dir}: {e}")
            return None

    if ".ostium.json" not in filepath:
        raise ValueError("filepath must end with .ostium.json")

    # Create the dictionary
    blueprint_dict = create_blueprint_dict(
        image_id, image_filename, pos_dict, affine_ijk_to_ras, affine_ras_to_ijk, comment, other
    )

    # Save to file
    try:
        with open(filepath, 'w') as f:
            json.dump(blueprint_dict, f, indent=4)
        if verbose:
            print(f"Successfully saved blueprint JSON to {filepath}")
    except IOError as e:
        print(f"I/O error occurred while saving {filepath}: {e}")
        return None
        
    return blueprint_dict



def save_slicer_json(filepath, pos_dict):
    """
    Takes a simple dictionary with RAS coordinates and saves it as a 3D Slicer .mkr.json file.
    """
    # Check and create parent directory
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        
    r_pos = pos_dict["R"]["ras"]
    l_pos = pos_dict["L"]["ras"]

    slicer_dict = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "RAS",
                "coordinateUnits": "mm",
                "controlPoints": [
                    {
                        "id": "1",
                        "label": "R",
                        "position": r_pos,
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "positionStatus": "defined"
                    },
                    {
                        "id": "2",
                        "label": "L",
                        "position": l_pos,
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "positionStatus": "defined"
                    }
                ],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 5.0,
                    "textScale": 3.0,
                    "pointLabelsVisibility": True,
                    "color": [0.8, 0.1, 0.1]
                }
            }
        ]
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(slicer_dict, f, indent=4)
    except IOError as e:
        print(f"I/O error occurred while saving Slicer markup {filepath}: {e}")

def load_slicer_json(filepath):
    """
    Loads a Slicer .mkr.json file. Transforms LPS to RAS if needed.
    Returns a simplified dictionary: { "R": {"ras": [...]}, "L": {"ras": [...]} }
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except IOError as e:
        print(f"Error reading {filepath}: {e}")
        return None

    markup = data.get("markups", [])[0]
    coord_sys = markup.get("coordinateSystem", "LPS")
    
    simple_pos_dict = {"L": {"ras": []}, "R": {"ras": []}}
    
    for pt in markup.get("controlPoints", []):
        label = pt.get("label", "").upper()
        if label in ["L", "R"]:
            pos = numpy.array(pt.get("position", [0.0, 0.0, 0.0]))
            
            # If coordinates are in LPS, convert to RAS: [-L, -P, S]
            if coord_sys == "LPS":
                pos[0] = -pos[0]
                pos[1] = -pos[1]
                
            simple_pos_dict[label]["ras"] = pos.tolist()
            
    return simple_pos_dict

def compute_image_metadata(image_path, simple_pos_dict, verbose: bool = False):
    """
    Given an image and a simple RAS position dictionary, opens the image with SITK
    and MONAI, and returns the populated position dictionary and affine matrices.
    verbose is passed to load_image_ras (notice when the SimpleITK fallback is used).
    """
    # ==========================
    # 1. SimpleITK Processing
    # ==========================
    img_sitk = sitk.ReadImage(image_path)
    direction = numpy.array(img_sitk.GetDirection()).reshape(3, 3)
    spacing = numpy.diag(img_sitk.GetSpacing())
    origin = numpy.array(img_sitk.GetOrigin()) # SITK natively uses LPS
    
    # Calculate IJK -> LPS Affine
    affine_sitk_lps = numpy.eye(4)
    affine_sitk_lps[:3, :3] = direction @ spacing
    affine_sitk_lps[:3, 3] = origin
    
    # Calculate IJK -> RAS Affine for SITK (flip X and Y)
    flip_lps_ras = numpy.diag([-1, -1, 1, 1])
    affine_sitk_ras = flip_lps_ras @ affine_sitk_lps
    affine_ras_sitk = numpy.linalg.inv(affine_sitk_ras)

    # ==========================
    # 2. MONAI Processing
    # ==========================
    # Load and reorient to standard RAS (MONAI, SimpleITK fallback: see load_image_ras)
    img_monai = load_image_ras(image_path, verbose=verbose)

    # MONAI's affine is already IJK -> RAS mapped to the newly oriented grid
    affine_monai_ras = img_monai.affine.detach().cpu().numpy().astype(float)
    affine_ras_monai = numpy.linalg.inv(affine_monai_ras)

    # ==========================
    # 3. Populate full dictionary
    # ==========================
    full_pos_dict = {}
    for key in ['L', 'R']:
        ras_pt = simple_pos_dict[key]['ras']
        
        # SITK: transform needs LPS coordinates
        lps_pt = [-ras_pt[0], -ras_pt[1], ras_pt[2]]
        ijk_sitk = img_sitk.TransformPhysicalPointToContinuousIndex(lps_pt)
        
        # MONAI: transform using standard matrix multiplication
        ras_pt_homog = numpy.array([ras_pt[0], ras_pt[1], ras_pt[2], 1.0])
        ijk_monai_homog = affine_ras_monai @ ras_pt_homog
        ijk_monai = ijk_monai_homog[:3].tolist()
        
        full_pos_dict[key] = {
            "ras": ras_pt,
            "ijk": {
                "sitk": list(ijk_sitk),
                "monai-ras": list(ijk_monai)
            }
        }
        
    affine_ijk_to_ras = {
        "sitk": affine_sitk_ras.tolist(),
        "monai-ras": affine_monai_ras.tolist()
    }
    
    affine_ras_to_ijk = {
        "sitk": affine_ras_sitk.tolist(),
        "monai-ras": affine_ras_monai.tolist()
    }
    
    return full_pos_dict, affine_ijk_to_ras, affine_ras_to_ijk


def read_image_header_lps(image_path):
    """
    Reads only the header of an image with SimpleITK (no voxel data).
    Returns (origin_lps (3,), spacing (3,), direction (3, 3) with columns = axis directions, size (3,)).
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    reader.ReadImageInformation()
    origin = numpy.array(reader.GetOrigin(), dtype=float)
    spacing = numpy.array(reader.GetSpacing(), dtype=float)
    direction = numpy.array(reader.GetDirection(), dtype=float).reshape(3, 3)
    size = numpy.array(reader.GetSize(), dtype=int)
    return origin, spacing, direction, size


def asoca_raw_point_to_ras(point, origin_lps, spacing, direction=None):
    """
    Converts an ASOCA raw ostium point (the 'ostia_coordinates' LEFT/RIGHT values, copied
    as 'world_mm_original_lps' in the *_ijk_ras_json files) to RAS world coordinates (mm).

    The raw points are neither LPS nor RAS world coordinates. They live in the frame of a
    VTK image built from the RAS-converted NRRD header with the direction matrix DROPPED:
        raw = (-ox, -oy, +oz) + ijk * spacing          (ox, oy, oz) = NRRD origin, LPS
    Checked on Diseased_1: raw R = (179.37, 262.19, -82.82) -> ijk = (233.2, 253.1, 163.7),
    which matches the verified 'coordinate_ijk' (233, 253, 164); reading raw as LPS world
    instead gives ijk = (694, 1103, 164), outside the 512 x 512 image.

    Only valid for axis-aligned images with identity direction (LPS), as the ASOCA NRRDs are:
    any other direction raises ValueError instead of returning a silently wrong point.

    Returns (ras (3,), ijk (3,) continuous voxel index in the native image grid).
    """
    point = numpy.asarray(point, dtype=float).reshape(3)
    origin_lps = numpy.asarray(origin_lps, dtype=float).reshape(3)
    spacing = numpy.abs(numpy.asarray(spacing, dtype=float).reshape(3))
    if direction is not None:
        direction = numpy.asarray(direction, dtype=float).reshape(3, 3)
        if not numpy.allclose(direction, numpy.eye(3), atol=1e-4):
            raise ValueError(f"ASOCA raw-point conversion needs an identity direction matrix, got {direction.tolist()}")
    vtk_origin = numpy.array([-origin_lps[0], -origin_lps[1], origin_lps[2]])
    ijk = (point - vtk_origin) / spacing
    lps = origin_lps + ijk * spacing          # identity direction
    ras = numpy.array([-lps[0], -lps[1], lps[2]])
    return ras, ijk


def asoca_header_from_json(data):
    """(origin_lps, spacing, direction) from the 'origin'/'spacing'/'orientation' fields of an
    ASOCA *_ijk_ras_json file ('orientation' = NRRD space directions, one row per axis)."""
    if not all(k in data for k in ("origin", "spacing")):
        return None
    origin = numpy.asarray(data["origin"], dtype=float)
    spacing = numpy.asarray(data["spacing"], dtype=float)
    direction = None
    if len(data.get("orientation", [])) == 9:
        space_dirs = numpy.asarray(data["orientation"], dtype=float).reshape(3, 3)
        direction = (space_dirs / numpy.linalg.norm(space_dirs, axis=1, keepdims=True)).T
    return origin, spacing, direction


def _asoca_header(data, image_path, filepath):
    """
    Header (origin_lps, spacing, direction) used to convert ASOCA raw points: from the image
    itself when image_path is given (authoritative), else from the fields copied in the JSON.
    Warns if both are available and disagree.
    """
    json_header = asoca_header_from_json(data)
    if image_path:
        origin, spacing, direction, _ = read_image_header_lps(image_path)
        if json_header is not None and not (
            numpy.allclose(json_header[0], origin, atol=1e-3) and numpy.allclose(numpy.abs(json_header[1]), spacing, atol=1e-5)
        ):
            print(f"WARNING: header stored in {filepath} differs from image {image_path}; using the image header.")
        return origin, spacing, direction
    if json_header is None:
        print(f"Cannot convert ASOCA points in {filepath}: pass the image path (the file has no header fields).")
    return json_header


def extract_simplified_ras(filepath, image_path=None):
    """
    Auto-detects the JSON structure of a given file (Slicer markups, custom ostia JSON,
    ASOCA JSON, or blueprint JSON) and extracts the simplified RAS coordinate dictionary:
    { "R": {"ras": [x, y, z]}, "L": {"ras": [x, y, z]} }

    image_path (optional): the image the markers belong to. Needed for the raw ASOCA format
    ('ostia_coordinates'), whose points can only be converted with the image header. For
    the ASOCA *_ijk_ras_json format it is preferred over the header copied in the JSON.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except IOError as e:
        print(f"Error reading {filepath}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {filepath}: {e}")
        return None

    simple_pos = {"R": {"ras": []}, "L": {"ras": []}}

    # 1. Format: 3D Slicer (.mrk.json or .mkr.json)
    if "markups" in data:
        # Delegate to the existing slicer loader which handles LPS -> RAS conversion
        return load_slicer_json(filepath)
        
    # 2. Format: Custom Ostia JSON (e.g., 1_final_ostia.json, CAT08 datasetXX_labels.json,
    #    ASOCA *_ijk_ras_json/<id>.json)
    elif "ostia" in data:
        header = None
        for key in ["R", "L"]:
            side = data["ostia"].get(key) or {}
            if len(side.get("coordinate_ras") or []) == 3:
                # ImageCAS, CAT08: already RAS world coordinates
                simple_pos[key]["ras"] = [float(c) for c in side["coordinate_ras"]]
            elif len(side.get("world_mm_original_lps") or []) == 3:
                # ASOCA: 'coordinate_ras' is empty, the raw ASOCA point sits in 'world_mm_original_lps'
                if header is None:
                    header = _asoca_header(data, image_path, filepath)
                    if header is None:
                        return None
                try:
                    ras, _ = asoca_raw_point_to_ras(side["world_mm_original_lps"], *header)
                except ValueError as e:
                    print(f"Cannot convert ASOCA point in {filepath}: {e}")
                    return None
                simple_pos[key]["ras"] = ras.tolist()
        return simple_pos

    # 3. Format: raw ASOCA ostium annotation ({"image_file": ..., "ostia_coordinates": {"LEFT": ..., "RIGHT": ...}})
    elif "ostia_coordinates" in data:
        header = _asoca_header(data, image_path, filepath)
        if header is None:
            return None
        for key, asoca_key in [("R", "RIGHT"), ("L", "LEFT")]:
            point = data["ostia_coordinates"].get(asoca_key)
            if point is None or len(point) != 3:
                continue
            try:
                ras, _ = asoca_raw_point_to_ras(point, *header)
            except ValueError as e:
                print(f"Cannot convert ASOCA point in {filepath}: {e}")
                return None
            simple_pos[key]["ras"] = ras.tolist()
        return simple_pos

    # 4. Format: Target Blueprint JSON (in case you pass a file already processed)
    elif "pos" in data:
        for key in ["R", "L"]:
            if key in data["pos"] and "ras" in data["pos"][key]:
                simple_pos[key]["ras"] = data["pos"][key]["ras"]
        return simple_pos

    else:
        print(f"Unrecognized JSON structure in {filepath}. Could not extract RAS coordinates.")
        return None


def asoca_raw_points_to_ras(points, origin_lps, spacing, direction=None):
    """Vectorised asoca_raw_point_to_ras(): (N, 3) raw ASOCA points -> (N, 3) RAS mm.
    Same frame, same identity-direction requirement (see asoca_raw_point_to_ras)."""
    points = numpy.asarray(points, dtype=float).reshape(-1, 3)
    origin_lps = numpy.asarray(origin_lps, dtype=float).reshape(3)
    if direction is not None:
        direction = numpy.asarray(direction, dtype=float).reshape(3, 3)
        if not numpy.allclose(direction, numpy.eye(3), atol=1e-4):
            raise ValueError(f"ASOCA raw-point conversion needs an identity direction matrix, got {direction.tolist()}")
    vtk_origin = numpy.array([-origin_lps[0], -origin_lps[1], origin_lps[2]])
    lps = origin_lps + (points - vtk_origin)  # ijk * spacing = points - vtk_origin
    return lps * numpy.array([-1.0, -1.0, 1.0])


# =====================================================================================
# Centerline graphs: GML, VTK polylines, 3D Slicer open curves
#
# Every reader returns the same plain networkx.Graph:
#   node attributes: 'pos' (numpy (3,), mm, in the FILE's own frame: nothing is
#                    converted except Slicer LPS markups, which are turned into RAS),
#                    'r' (float mm, nan if unknown), 'side' ('L', 'R', 'RL' or None),
#                    'is_ostium' (bool), 'line_start' (int: polylines starting here)
#   edge attribute:  'length' (mm)
#   graph attributes: 'source' (path), 'format' ('gml', 'vtk', 'slicer'), 'frame_hint'
#                    ('ras' for GML and Slicer, 'unknown' for VTK, which has no frame)
# =====================================================================================

_RADIUS_ARRAY_NAMES = ("maximuminscribedsphereradius", "radius", "radii", "r")


def _nearest_neighbours(query, reference):
    """(distance, index) of the nearest reference point for every query point:
    scipy cKDTree when available, else chunked brute force (numpy only)."""
    query = numpy.asarray(query, dtype=float).reshape(-1, 3)
    reference = numpy.asarray(reference, dtype=float).reshape(-1, 3)
    if len(reference) == 0:
        return numpy.full(len(query), numpy.inf), numpy.full(len(query), -1, dtype=int)
    try:
        from scipy.spatial import cKDTree
        distance, index = cKDTree(reference).query(query)
        return numpy.asarray(distance, dtype=float), numpy.asarray(index, dtype=int)
    except ImportError:
        distance = numpy.empty(len(query))
        index = numpy.empty(len(query), dtype=int)
        step = max(1, int(2e7 // max(len(reference), 1)))
        for lo in range(0, len(query), step):
            d2 = ((query[lo:lo + step, None, :] - reference[None, :, :]) ** 2).sum(-1)
            index[lo:lo + step] = d2.argmin(1)
            distance[lo:lo + step] = numpy.sqrt(d2[numpy.arange(len(d2)), index[lo:lo + step]])
        return distance, index


def _side_from_value(value):
    """'L'/'R'/'RL'/None from any side encoding: 'L', 'left', 'ArteryNodeSide.LEFT',
    an ArteryNodeSide enum, or its integer value (resolved through hcatnetwork)."""
    if value is None:
        return None
    if hasattr(value, "name"):
        value = value.name
    if isinstance(value, (int, numpy.integer)) or (isinstance(value, str) and value.strip().lstrip("-").isdigit()):
        try:
            from hcatnetwork.node.node import ArteryNodeSide
            value = ArteryNodeSide(int(value)).name
        except Exception:
            return None
    text = str(value).strip().upper().split(".")[-1]
    return {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R", "RL": "RL", "LR": "RL", "BOTH": "RL"}.get(text)


def _is_ostium_value(value) -> bool:
    if value is None:
        return False
    if hasattr(value, "name"):
        value = value.name
    if isinstance(value, (int, numpy.integer)) or (isinstance(value, str) and value.strip().isdigit()):
        try:
            from hcatnetwork.node.node import ArteryNodeTopology
            value = ArteryNodeTopology(int(value)).name
        except Exception:
            return False
    return str(value).strip().upper().split(".")[-1] == "OSTIUM"


def _finish_graph(graph, source, fmt, frame_hint):
    import networkx
    for u, v, data in graph.edges(data=True):
        data["length"] = float(numpy.linalg.norm(graph.nodes[u]["pos"] - graph.nodes[v]["pos"]))
    graph.remove_edges_from(list(networkx.selfloop_edges(graph)))
    graph.graph.update(source=str(source), format=fmt, frame_hint=frame_hint)
    return graph


def read_centerline_gml(path):
    """A GML centerline graph. Our own files (HCATNetwork SimpleCenterlineGraph) are
    read with HCATNetwork when it is installed; anything else with networkx. Node
    positions come from 'x','y','z' (or a 3-element 'pos'), radius from 'r'/'radius',
    side from 'side', ostia from 'topology' == OSTIUM."""
    import networkx
    try:
        from hcatnetwork.io.io import load_graph
        from hcatnetwork.graph.graph import SimpleCenterlineGraph
        source = load_graph(path, output_type=SimpleCenterlineGraph)
    except Exception:
        source = networkx.read_gml(path, label="id", destringizer=None)
    graph = networkx.Graph()
    for node_id, data in source.nodes(data=True):
        if all(k in data for k in ("x", "y", "z")):
            pos = [data["x"], data["y"], data["z"]]
        elif "pos" in data and len(data["pos"]) == 3:
            pos = data["pos"]
        else:
            raise ValueError(f"{path}: node {node_id} has no position ('x','y','z' or 'pos')")
        radius = data.get("r", data.get("radius", numpy.nan))
        graph.add_node(
            node_id,
            pos=numpy.asarray([float(c) for c in pos], dtype=float),
            r=float(radius) if radius is not None else numpy.nan,
            side=_side_from_value(data.get("side")),
            is_ostium=_is_ostium_value(data.get("topology")),
            line_start=0,
            is_start=False,
            segment=numpy.nan,
        )
    graph.add_edges_from(source.edges())
    return _finish_graph(graph, path, "gml", "ras")


def _read_vtk_polylines_with_vtk(path):
    """(points (N, 3), point arrays {name: (N,) or (N, k)}, line cells [ids]) with the vtk package."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    lower = str(path).lower()
    if lower.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllFieldsOn()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"{path}: no points read (not a PolyData file?)")
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    arrays = {}
    for data in (poly.GetPointData(), poly.GetFieldData()):
        for i in range(data.GetNumberOfArrays()):
            array = data.GetAbstractArray(i)
            if array is None or not array.GetName() or array.GetNumberOfTuples() != len(points):
                continue
            if isinstance(array, vtk.vtkStringArray):  # e.g. ImageCAS-X segment_name
                arrays[array.GetName()] = numpy.array([array.GetValue(k) for k in range(len(points))], dtype=object)
            elif data.GetArray(i) is not None:
                arrays[array.GetName()] = vtk_to_numpy(data.GetArray(i)).astype(float)
    lines = []
    cells = poly.GetLines()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()
    while cells.GetNextCell(id_list):
        lines.append([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])
    return points, arrays, lines


_LEGACY_TYPES = {"float": "f4", "double": "f8", "int": "i4", "unsigned_int": "u4", "long": "i8",
                 "unsigned_long": "u8", "vtktypeint64": "i8", "vtktypeuint64": "u8", "short": "i2",
                 "unsigned_short": "u2", "char": "i1", "unsigned_char": "u1", "vtkidtype": "i4", "bit": "u1"}


def _read_vtk_legacy_python(path):
    """Legacy .vtk POLYDATA reader, ASCII or BINARY (big-endian), old (cell-size-prefixed
    LINES) or version >= 5 (OFFSETS / CONNECTIVITY) layout: POINTS, LINES, and the
    numeric POINT_DATA SCALARS / FIELD arrays (string arrays are skipped)."""
    import re
    data = open(path, "rb").read()
    binary = b"\nBINARY" in data[:300]
    cursor = 0

    def next_line():
        nonlocal cursor
        while True:
            end = data.index(b"\n", cursor)
            line = data[cursor:end].decode("latin-1").strip()
            cursor = end + 1
            if line:
                return line

    def read_values(count, vtk_type):
        nonlocal cursor
        if binary:
            dtype = numpy.dtype(">" + _LEGACY_TYPES[vtk_type.lower()])
            values = numpy.frombuffer(data, dtype=dtype, count=count, offset=cursor)
            cursor += count * dtype.itemsize
            return values.astype(float if dtype.kind == "f" else numpy.int64)
        values = []
        while len(values) < count:
            values.extend(next_line().split())
        return numpy.array(values[:count], dtype=float)

    def read_strings(count):
        """Legacy string arrays. BINARY: each string is preceded by its length, whose
        first byte's two top bits give the size of the length field (11: 6 bits in 1
        byte, 10: 14 bits in 2 bytes, 01: 30 bits in 4 bytes, 00: 62 bits in 8 bytes,
        big-endian). ASCII: one string per line, %XX-escaped."""
        nonlocal cursor
        out = []
        if not binary:
            from urllib.parse import unquote
            for _ in range(count):
                end = data.index(b"\n", cursor)
                out.append(unquote(data[cursor:end].decode("latin-1")))
                cursor = end + 1
            return out
        for _ in range(count):
            first = data[cursor]
            kind = first >> 6
            size = {3: 1, 2: 2, 1: 4, 0: 8}[kind]
            length = int.from_bytes(data[cursor:cursor + size], "big") & ((1 << (8 * size - 2)) - 1)
            cursor += size
            out.append(data[cursor:cursor + length].decode("utf-8", errors="replace"))
            cursor += length
        return out

    for _ in range(4):  # version, title, ASCII/BINARY, DATASET
        next_line()
    points, arrays, lines, n_points = None, {}, [], 0
    field_header = re.compile(rb"\n([^\s]+) (\d+) (\d+) (" + "|".join(_LEGACY_TYPES).encode() + rb")\r?\n")
    while cursor < len(data):
        try:
            words = next_line().split()
        except ValueError:
            break
        keyword = words[0].upper()
        if keyword == "POINTS":
            n_points = int(words[1])
            points = read_values(3 * n_points, words[2]).reshape(n_points, 3).astype(float)
        elif keyword in ("LINES", "VERTICES", "POLYGONS", "TRIANGLE_STRIPS"):
            first, second = int(words[1]), int(words[2])
            peek = data[cursor:cursor + 40].lstrip()
            if peek.startswith(b"OFFSETS"):
                offsets = read_values(first, next_line().split()[1]).astype(int)
                connectivity = read_values(second, next_line().split()[1]).astype(int)
                cells = [connectivity[offsets[k]:offsets[k + 1]].tolist() for k in range(first - 1)]
            else:
                values = read_values(second, "int").astype(int)
                cells, j = [], 0
                while j < len(values):
                    cells.append(values[j + 1:j + 1 + values[j]].tolist())
                    j += 1 + values[j]
            if keyword == "LINES":
                lines = cells
        elif keyword == "POINT_DATA":
            n_points = int(words[1])
        elif keyword == "CELL_DATA":
            break  # nothing needed after point data
        elif keyword == "SCALARS":
            n_components = int(words[3]) if len(words) > 3 else 1
            if data[cursor:cursor + 40].lstrip().startswith(b"LOOKUP_TABLE"):
                next_line()
            arrays[words[1]] = read_values(n_points * n_components, words[2]).reshape(n_points, n_components).squeeze()
        elif keyword == "FIELD":
            for _ in range(int(words[2])):
                name, n_components, n_tuples, vtk_type = next_line().split()[:4]
                if vtk_type.lower() == "string":
                    count = int(n_components) * int(n_tuples)
                    try:
                        strings = read_strings(count)
                    except Exception:
                        strings = None
                    if strings is not None and int(n_tuples) == n_points:
                        arrays[name] = numpy.array(strings, dtype=object)
                    else:  # unreadable: jump to the next array header
                        match = field_header.search(data, cursor)
                        if match is None:
                            cursor = len(data)
                            break
                        cursor = match.start() + 1
                    continue
                values = read_values(int(n_components) * int(n_tuples), vtk_type)
                if int(n_tuples) == n_points:
                    arrays[name] = values.reshape(n_points, int(n_components)).squeeze()
        elif keyword in ("METADATA", "INFORMATION", "NAME", "DATA", "LOOKUP_TABLE"):
            continue
    if points is None:
        raise ValueError(f"{path}: no POINTS section")
    return points, arrays, lines


def _read_vtp_python(path):
    """XML PolyData (.vtp) reader: inline DataArrays in ascii or base64 binary, raw or
    zlib-compressed (appended data is not supported). Returns points, numeric point arrays
    and line cells."""
    import base64
    import zlib
    import xml.etree.ElementTree as ElementTree
    root = ElementTree.parse(path).getroot()
    header_type = numpy.dtype(("<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">")
                              + {"UInt32": "u4", "UInt64": "u8"}[root.get("header_type", "UInt32")])
    endian = header_type.str[0]
    compressed = root.get("compressor") is not None
    types = {"Float32": "f4", "Float64": "f8", "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2",
             "Int32": "i4", "UInt32": "u4", "Int64": "i8", "UInt64": "u8"}

    def decode(element):
        dtype = numpy.dtype(endian + types[element.get("type")])
        text = (element.text or "").strip()
        fmt = element.get("format", "ascii")
        if fmt == "ascii":
            return numpy.array(text.split(), dtype=dtype)
        if fmt != "binary":
            raise ValueError(f"{path}: DataArray format '{fmt}' not supported without the vtk package")
        size = header_type.itemsize
        if not compressed:
            raw = base64.b64decode(text)
            n_bytes = int(numpy.frombuffer(raw[:size], header_type)[0])
            return numpy.frombuffer(raw[size:size + n_bytes], dtype)
        first = base64.b64decode(text[: (3 * size + 2) // 3 * 4])
        n_blocks = int(numpy.frombuffer(first[:size], header_type)[0])
        header_chars = ((3 + n_blocks) * size + 2) // 3 * 4
        header = numpy.frombuffer(base64.b64decode(text[:header_chars])[: (3 + n_blocks) * size], header_type)
        payload = base64.b64decode(text[header_chars:])
        out, offset = [], 0
        for compressed_size in header[3:]:
            out.append(zlib.decompress(payload[offset:offset + int(compressed_size)]))
            offset += int(compressed_size)
        return numpy.frombuffer(b"".join(out), dtype)

    piece = root.find("PolyData/Piece")
    n_points = int(piece.get("NumberOfPoints"))
    points = decode(piece.find("Points/DataArray")).astype(float).reshape(n_points, 3)
    arrays = {}
    for element in piece.findall("PointData/DataArray"):
        if element.get("type") in types:
            values = decode(element).astype(float)
            arrays[element.get("Name")] = values.reshape(n_points, -1).squeeze()
    lines = []
    lines_element = piece.find("Lines")
    if lines_element is not None:
        named = {e.get("Name"): e for e in lines_element.findall("DataArray")}
        connectivity = decode(named["connectivity"]).astype(int)
        offsets = decode(named["offsets"]).astype(int)
        start = 0
        for end in offsets:
            lines.append(connectivity[start:end].tolist())
            start = end
    return points, arrays, lines


def read_vtk_polylines(path):
    """List of polylines from a VTK PolyData file (.vtk legacy ASCII/binary, or .vtp
    XML): one dict per line cell, {'points': (N, 3), 'radius': (N,) or None, 'side',
    'start_flags': (N,) bool or None, 'name'}. Uses the vtk package when installed,
    else the pure-python readers above. A file with points but no line cells is read as
    one polyline in point order.
      - radius: a point array named MaximumInscribedSphereRadius (ASOCA / VMTK), Radius, r;
      - start_flags: a point array named start_points (ImageCAS-X marks the tree roots);
      - segment: a point array named segment_label (ImageCAS-X anatomical segment ids);
      - side: guessed from the file name ('left', 'RCA', ...), else None."""
    try:
        points, arrays, lines = _read_vtk_polylines_with_vtk(path)
    except ImportError:
        if str(path).lower().endswith(".vtp"):
            points, arrays, lines = _read_vtp_python(path)
        else:
            points, arrays, lines = _read_vtk_legacy_python(path)
    by_lower = {name.lower(): values for name, values in arrays.items()}
    radius = next((numpy.asarray(by_lower[n], dtype=float).reshape(len(points), -1)[:, 0]
                   for n in _RADIUS_ARRAY_NAMES if n in by_lower), None)
    start_flags = numpy.asarray(by_lower["start_points"]).reshape(len(points), -1)[:, 0] > 0 if "start_points" in by_lower else None
    segment = numpy.asarray(by_lower["segment_label"], dtype=float).reshape(len(points), -1)[:, 0] if "segment_label" in by_lower else None
    segment_names = {}
    if segment is not None and "segment_name" in by_lower:  # ImageCAS-X: label id -> name (majority over its points)
        names = by_lower["segment_name"]
        for label in numpy.unique(segment[numpy.isfinite(segment)]):
            values = list(names[segment == label])
            segment_names[str(int(label))] = max(set(values), key=values.count)
    if not lines:
        lines = [list(range(len(points)))]
    name = os.path.basename(str(path))
    side = _side_from_name(name)
    return [
        {"points": points[ids], "radius": None if radius is None else radius[ids], "side": side,
         "start_flags": None if start_flags is None else start_flags[ids],
         "segment": None if segment is None else segment[ids], "segment_names": segment_names,
         "name": f"{name}#{k}"}
        for k, ids in enumerate(lines) if len(ids) >= 2
    ]


def _side_from_name(name):
    import re
    text = name.lower()
    if re.search(r"(^|[^a-z])(left|lca|lm|lad|lcx)([^a-z]|$)", text):
        return "L"
    if re.search(r"(^|[^a-z])(right|rca)([^a-z]|$)", text):
        return "R"
    if re.search(r"(^|[_\-\s])l([_\-\s.]|$)", text):
        return "L"
    if re.search(r"(^|[_\-\s])r([_\-\s.]|$)", text):
        return "R"
    return None


def read_slicer_curves(path):
    """Open curves from 3D Slicer markups: a folder of *.SlicerOpenCurve.mkr.json (or
    any *.mrk.json / *.mkr.json holding curves), or a single such file. One polyline
    per curve markup, in RAS (LPS markups are converted). Side is guessed from the
    markup / file name ('left', 'LAD', '_L_', ...), else None."""
    import glob
    if os.path.isdir(path):
        files = sorted(set(glob.glob(os.path.join(path, "*.mkr.json")) + glob.glob(os.path.join(path, "*.mrk.json"))))
    else:
        files = [path]
    polylines = []
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
        for k, markup in enumerate(data.get("markups", [])):
            if "curve" not in str(markup.get("type", "")).lower() or "closed" in str(markup.get("type", "")).lower():
                continue
            points = numpy.array([p["position"] for p in markup.get("controlPoints", []) if "position" in p], dtype=float)
            if len(points) < 2:
                continue
            if markup.get("coordinateSystem", "LPS").upper() == "LPS":
                points = points * numpy.array([-1.0, -1.0, 1.0])
            name = markup.get("name") or os.path.basename(file_path)
            polylines.append({"points": points, "radius": None,
                              "side": _side_from_name(str(name)) or _side_from_name(os.path.basename(file_path)),
                              "name": f"{os.path.basename(file_path)}#{k}"})
    if not polylines:
        raise ValueError(f"{path}: no open curve markups found")
    return polylines


def _polyline_length(points):
    return float(numpy.linalg.norm(numpy.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _densify(points, max_step):
    """Points along the polyline with at most max_step between consecutive ones."""
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(numpy.ceil(numpy.linalg.norm(b - a) / max_step)))
        out.append(a + (b - a) * (numpy.arange(1, n + 1)[:, None] / n))
    return numpy.vstack(out)


def polylines_to_graph(polylines, merge_tol_mm=1e-3, overlap_tol_mm=0.0, source="", fmt="vtk", frame_hint="unknown"):
    """Builds one graph from polylines.

    merge_tol_mm: a point closer than this to a node of an earlier polyline is that node
    (and consecutive duplicates collapse), so polylines that share vertices (shared ids
    or duplicated coordinates) join into one tree.

    overlap_tol_mm > 0 also removes overlapping geometry: many centerline formats (ASOCA,
    VMTK) store one polyline per ostium-to-endpoint path, so proximal segments appear
    several times without sharing vertices. Polylines are added longest first; the parts
    of a new polyline lying within overlap_tol_mm of geometry already added are dropped,
    and each remaining run is attached to the nearest earlier node where it leaves (or
    re-enters) the existing geometry. Runs shorter than overlap_tol_mm bounded by overlap
    on both sides (jitter where two copies of a vessel diverge) are dropped too.

    'is_start' marks nodes flagged as tree roots by the file itself (ImageCAS-X's
    start_points); without such flags,
    'line_start' counts, per node, the polylines that start there (a polyline starting
    inside existing geometry credits the nearest earlier node): with ASOCA-style files,
    where every line starts at the ostium, the ostium is the node with the largest count.
    """
    import networkx
    graph = networkx.Graph()
    positions = []  # node id -> position (node ids are 0..n-1)
    dense_geometry = numpy.zeros((0, 3))

    def new_node(position, radius, side, segment=None):
        graph.add_node(len(positions), pos=numpy.asarray(position, dtype=float),
                       r=float(radius) if radius is not None else numpy.nan,
                       side=side, is_ostium=False, line_start=0, is_start=False,
                       segment=float(segment) if segment is not None else numpy.nan)
        positions.append(numpy.asarray(position, dtype=float))
        return len(positions) - 1

    def runs(mask):
        k = 0
        while k < len(mask):
            if mask[k]:
                end = k
                while end + 1 < len(mask) and mask[end + 1]:
                    end += 1
                yield k, end
                k = end + 1
            else:
                k += 1

    ordered = sorted(polylines, key=lambda p: -_polyline_length(p["points"])) if overlap_tol_mm > 0 else list(polylines)
    for polyline in ordered:
        points = numpy.asarray(polyline["points"], dtype=float)
        radius = polyline.get("radius")
        side = polyline.get("side")
        n_existing = len(positions)
        existing = numpy.array(positions[:n_existing]).reshape(-1, 3)
        to_existing, nearest_existing = _nearest_neighbours(points, existing)

        keep = numpy.ones(len(points), dtype=bool)
        if overlap_tol_mm > 0 and len(dense_geometry):
            to_geometry, _ = _nearest_neighbours(points, dense_geometry)
            keep = to_geometry > overlap_tol_mm
            for lo, hi in list(runs(keep)):
                if lo > 0 and hi < len(points) - 1 and _polyline_length(points[lo:hi + 1]) < overlap_tol_mm:
                    keep[lo:hi + 1] = False

        if not keep[0] and polyline.get("start_flags") is None and n_existing and to_existing[0] <= max(overlap_tol_mm, merge_tol_mm):
            graph.nodes[int(nearest_existing[0])]["line_start"] += 1

        for lo, hi in runs(keep):
            chain, chain_nodes = [], []
            for k in range(lo, hi + 1):
                if n_existing and to_existing[k] <= merge_tol_mm:
                    node = int(nearest_existing[k])
                elif chain and numpy.linalg.norm(positions[chain[-1]] - points[k]) <= merge_tol_mm:
                    node = chain[-1]
                else:
                    segment = polyline.get("segment")
                    node = new_node(points[k], None if radius is None else radius[k], side,
                                    None if segment is None else segment[k])
                if graph.nodes[node]["side"] is None:
                    graph.nodes[node]["side"] = side
                if chain and chain[-1] != node:
                    graph.add_edge(chain[-1], node)
                if not chain or chain[-1] != node:
                    chain.append(node)
                chain_nodes.append(node)
            if lo == 0 and polyline.get("start_flags") is None:
                graph.nodes[chain[0]]["line_start"] += 1
            flags = polyline.get("start_flags")
            if flags is not None:
                for k, node in zip(range(lo, hi + 1), chain_nodes):
                    if flags[k]:
                        graph.nodes[node]["is_start"] = True
            if n_existing and overlap_tol_mm > 0:
                if lo > 0 and int(nearest_existing[lo - 1]) != chain[0]:
                    graph.add_edge(int(nearest_existing[lo - 1]), chain[0])
                if hi < len(points) - 1 and int(nearest_existing[hi + 1]) != chain[-1]:
                    graph.add_edge(chain[-1], int(nearest_existing[hi + 1]))
        if overlap_tol_mm > 0:
            dense_geometry = numpy.vstack([dense_geometry, _densify(points, max(overlap_tol_mm / 2.0, 0.05))])
    return _finish_graph(graph, source, fmt, frame_hint)


def load_centerline_graph(path, fmt="auto", overlap_tol_mm=0.0, merge_tol_mm=1e-3):
    """Reads a centerline graph in any supported format into the common networkx form
    (see the section comment above):
      - 'gml'    : .gml file (our SimpleCenterlineGraph output, or any GML with x/y/z);
      - 'vtk'    : .vtk / .vtp PolyData with line cells (ASOCA, ImageCAS-X, VMTK);
      - 'slicer' : a folder of *.SlicerOpenCurve.mkr.json, or one markup file.
    fmt='auto' picks by extension / folder. overlap_tol_mm and merge_tol_mm apply to the
    polyline formats only (see polylines_to_graph)."""
    path = str(path)
    if fmt == "auto":
        lower = path.lower()
        if os.path.isdir(path) or lower.endswith((".mkr.json", ".mrk.json")):
            fmt = "slicer"
        elif lower.endswith(".gml"):
            fmt = "gml"
        elif lower.endswith((".vtk", ".vtp")):
            fmt = "vtk"
        else:
            raise ValueError(f"Cannot tell the centerline format of {path}: pass fmt explicitly")
    if fmt == "gml":
        return read_centerline_gml(path)
    if fmt == "vtk":
        polylines = read_vtk_polylines(path)
        graph = polylines_to_graph(polylines, merge_tol_mm, overlap_tol_mm, path, "vtk", "unknown")
        graph.graph["segment_names"] = polylines[0].get("segment_names", {}) if polylines else {}
        return graph
    if fmt == "slicer":
        return polylines_to_graph(read_slicer_curves(path), merge_tol_mm, overlap_tol_mm, path, "slicer", "ras")
    raise ValueError(f"Unknown centerline format '{fmt}'")
