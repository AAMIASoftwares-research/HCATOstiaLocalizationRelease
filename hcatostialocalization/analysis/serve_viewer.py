from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from hcatostialocalization.analysis.volume_io import normalize_uint8, read_volume


def _float_or_none(value):
    if value in ("", None):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _point_from_record(record: dict, prefix: str) -> list[float] | None:
    vals = [_float_or_none(record.get(f"{prefix}_{axis}")) for axis in ("z", "y", "x")]
    return vals if all(v is not None for v in vals) else None


def _case_key(record: dict) -> str:
    return str(record.get("case_id") or record.get("image_path") or record.get("heatmap_path") or "case")


def write_viewer_files(output_dir: Path, records: list[dict], summary: dict | None = None) -> None:
    viewer_dir = output_dir / "viewer"
    assets_dir = viewer_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict] = {}
    for i, record in enumerate(records):
        key = _case_key(record)
        case = grouped.setdefault(
            key,
            {
                "case_id": key,
                "dataset": record.get("dataset", ""),
                "image_path": record.get("image_path", ""),
                "shape": [
                    _float_or_none(record.get("shape_z")),
                    _float_or_none(record.get("shape_y")),
                    _float_or_none(record.get("shape_x")),
                ],
                "spacing_zyx": [
                    _float_or_none(record.get("spacing_z")),
                    _float_or_none(record.get("spacing_y")),
                    _float_or_none(record.get("spacing_x")),
                ],
                "records": [],
            },
        )
        side = str(record.get("side") or "").upper()
        case["records"].append(
            {
                "record_index": i,
                "side": side,
                "is_outlier": bool(record.get("is_outlier")),
                "distance_to_label_mm": _float_or_none(record.get("distance_to_label_mm")),
                "label": _point_from_record(record, "gt"),
                "prediction": _point_from_record(record, "pred"),
                "heatmap_path": record.get("heatmap_path", ""),
                "heatmap_entropy_norm": _float_or_none(record.get("heatmap_entropy_norm")),
                "heatmap_spread_mm": _float_or_none(record.get("heatmap_spread_mm")),
                "heatmap_peak_prominence_p95": _float_or_none(record.get("heatmap_peak_prominence_p95")),
                "heatmap_peak_prominence_median": _float_or_none(record.get("heatmap_peak_prominence_median")),
            }
        )

    index = {
        "cases": list(grouped.values()),
        "summary": summary or {},
    }
    (viewer_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (viewer_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (assets_dir / "viewer.css").write_text(VIEWER_CSS, encoding="utf-8")
    (assets_dir / "viewer.js").write_text(VIEWER_JS, encoding="utf-8")


class ViewerHandler(BaseHTTPRequestHandler):
    output_dir: Path
    viewer_dir: Path
    index_data: dict
    volume_cache: OrderedDict
    render_cache: OrderedDict
    volume_cache_lock: threading.RLock
    render_cache_lock: threading.RLock
    max_cached_volumes: int
    max_cached_render_volumes: int

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        raw = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_slice_raw(self, payload: dict, status: int = 200) -> None:
        raw = payload["data"]
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Slice-Width", str(payload["width"]))
        self.send_header("X-Slice-Height", str(payload["height"]))
        self.send_header("X-Shape-Z", str(payload["shape"][0]))
        self.send_header("X-Shape-Y", str(payload["shape"][1]))
        self.send_header("X-Shape-X", str(payload["shape"][2]))
        self.send_header("X-Spacing-Z", str(payload["spacing_zyx"][0]))
        self.send_header("X-Spacing-Y", str(payload["spacing_zyx"][1]))
        self.send_header("X-Spacing-X", str(payload["spacing_zyx"][2]))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path in {"/", "/index.html"}:
                self._send_file(self.viewer_dir / "index.html")
            elif path == "/assets/viewer.css":
                self._send_file(self.viewer_dir / "assets" / "viewer.css")
            elif path == "/assets/viewer.js":
                self._send_file(self.viewer_dir / "assets" / "viewer.js")
            elif path == "/api/index":
                self._send_json(self.index_data)
            elif path == "/api/slice":
                self._send_json(self._api_slice(query))
            elif path == "/api/slice_raw":
                self._send_slice_raw(self._api_slice_raw(query))
            elif path == "/api/heatmap3d":
                self._send_json(self._api_heatmap3d(query))
            else:
                self.send_error(404)
        except Exception as exc:  # noqa: BLE001 - HTTP API should report errors as JSON.
            self._send_json({"error": str(exc)}, status=500)

    def _record(self, query: dict) -> tuple[dict, dict]:
        case_idx = int(query.get("case", ["0"])[0])
        side = query.get("side", [""])[0].upper()
        cases = self.index_data["cases"]
        case = cases[case_idx]
        records = case["records"]
        record = next((r for r in records if r.get("side") == side), records[0] if records else {})
        return case, record

    def _load_source(self, case: dict, record: dict, source: str) -> tuple[np.ndarray | None, tuple[float, float, float] | None]:
        if source == "image":
            path = case.get("image_path")
        elif source == "heatmap":
            path = record.get("heatmap_path")
        else:
            raise ValueError(f"unknown source: {source}")
        if not path:
            return None, None
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)
        return self._load_cached_volume(path_obj)

    def _load_cached_volume(self, path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
        path = path.resolve()
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self.volume_cache_lock:
            cached = self.volume_cache.get(key)
            if cached is not None:
                self.volume_cache.move_to_end(key)
                return cached

            loaded = read_volume(path)
            self.volume_cache[key] = loaded
            self.volume_cache.move_to_end(key)
            while len(self.volume_cache) > self.max_cached_volumes:
                self.volume_cache.popitem(last=False)
            return loaded

    def _load_render_source(self, case: dict, record: dict, source: str) -> dict:
        if source == "image":
            path = case.get("image_path")
        elif source == "heatmap":
            path = record.get("heatmap_path")
        else:
            raise ValueError(f"unknown source: {source}")

        if not path:
            shape = [int(v) for v in case.get("shape") or [64, 64, 64] if v is not None]
            if len(shape) != 3:
                shape = [64, 64, 64]
            spacing = tuple(float(v or 1.0) for v in case.get("spacing_zyx") or [1.0, 1.0, 1.0])
            return {
                "volume_u8": np.zeros(shape, dtype=np.uint8),
                "shape": tuple(shape),
                "spacing": spacing,
                "slice_cache": {},
                "warming": True,
            }

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)

        path_obj = path_obj.resolve()
        stat = path_obj.stat()
        key = (str(path_obj), stat.st_mtime_ns, stat.st_size, "render")
        with self.render_cache_lock:
            cached = self.render_cache.get(key)
            if cached is not None:
                self.render_cache.move_to_end(key)
                return cached

        volume, spacing = self._load_cached_volume(path_obj)
        volume_u8 = normalize_uint8(volume)
        entry = {
            "volume_u8": volume_u8,
            "shape": tuple(int(v) for v in volume_u8.shape[:3]),
            "spacing": tuple(float(v) for v in (spacing or (1.0, 1.0, 1.0))),
            "slice_cache": {},
            "warming": False,
        }
        with self.render_cache_lock:
            cached = self.render_cache.get(key)
            if cached is not None:
                self.render_cache.move_to_end(key)
                return cached
            self.render_cache[key] = entry
            self.render_cache.move_to_end(key)
            while len(self.render_cache) > self.max_cached_render_volumes:
                self.render_cache.popitem(last=False)

        self._start_slice_warmup(key, entry)
        return entry

    def _start_slice_warmup(self, key: tuple, entry: dict) -> None:
        with self.render_cache_lock:
            if entry.get("warming"):
                return
            entry["warming"] = True

        def warm() -> None:
            try:
                volume_u8 = entry["volume_u8"]
                for plane, count in (
                    ("z", volume_u8.shape[0]),
                    ("y", volume_u8.shape[1]),
                    ("x", volume_u8.shape[2]),
                ):
                    for index in range(count):
                        with self.render_cache_lock:
                            if key not in self.render_cache:
                                return
                            slice_cache = entry["slice_cache"]
                            if (plane, index) in slice_cache:
                                continue
                        self._slice_from_entry(entry, plane, index)
            finally:
                with self.render_cache_lock:
                    entry["warming"] = False

        threading.Thread(target=warm, daemon=True).start()

    def _slice_from_entry(self, entry: dict, plane: str, index: int) -> dict:
        volume_u8 = entry["volume_u8"]
        if plane == "z":
            index = max(0, min(index, volume_u8.shape[0] - 1))
            sl = volume_u8[index, :, :]
        elif plane == "y":
            index = max(0, min(index, volume_u8.shape[1] - 1))
            sl = volume_u8[:, index, :]
        elif plane == "x":
            index = max(0, min(index, volume_u8.shape[2] - 1))
            sl = volume_u8[:, :, index]
        else:
            raise ValueError(f"unknown plane: {plane}")

        cache_key = (plane, index)
        with self.render_cache_lock:
            cached = entry["slice_cache"].get(cache_key)
            if cached is not None:
                return cached

        sl = np.ascontiguousarray(sl)
        payload = {
            "width": int(sl.shape[1]),
            "height": int(sl.shape[0]),
            "shape": [int(v) for v in entry["shape"]],
            "spacing_zyx": [float(v) for v in entry["spacing"]],
            "data": sl.tobytes(),
        }
        with self.render_cache_lock:
            entry["slice_cache"][cache_key] = payload
        return payload

    def _api_slice(self, query: dict) -> dict:
        case, record = self._record(query)
        plane = query.get("plane", ["z"])[0].lower()
        index = int(query.get("index", ["0"])[0])
        source = query.get("source", ["image"])[0].lower()
        entry = self._load_render_source(case, record, source)
        payload = self._slice_from_entry(entry, plane, index)
        return {
            "width": payload["width"],
            "height": payload["height"],
            "shape": payload["shape"],
            "spacing_zyx": payload["spacing_zyx"],
            "data_b64": base64.b64encode(payload["data"]).decode("ascii"),
        }

    def _api_slice_raw(self, query: dict) -> dict:
        case, record = self._record(query)
        plane = query.get("plane", ["z"])[0].lower()
        index = int(query.get("index", ["0"])[0])
        source = query.get("source", ["image"])[0].lower()
        entry = self._load_render_source(case, record, source)
        return self._slice_from_entry(entry, plane, index)

    def _api_heatmap3d(self, query: dict) -> dict:
        case, record = self._record(query)
        threshold = float(query.get("threshold", ["0.20"])[0])
        max_points = int(query.get("max_points", ["8000"])[0])
        volume, _spacing = self._load_source(case, record, "heatmap")
        if volume is None:
            return {"points": []}

        h = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        h = h - float(np.min(h))
        max_value = float(np.max(h))
        if max_value <= 0:
            return {"points": []}
        h = h / max_value
        coords = np.argwhere(h >= threshold)
        if coords.shape[0] > max_points:
            values = h[coords[:, 0], coords[:, 1], coords[:, 2]]
            keep = np.argpartition(values, -max_points)[-max_points:]
            coords = coords[keep]
            values = values[keep]
        else:
            values = h[coords[:, 0], coords[:, 1], coords[:, 2]]
        return {
            "points": [
                [float(x), float(y), float(z), float(v)]
                for (z, y, x), v in zip(coords.tolist(), values.tolist(), strict=False)
            ],
            "shape": [int(v) for v in h.shape[:3]],
        }


def serve(output_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    output_dir = Path(output_dir).resolve()
    viewer_dir = output_dir / "viewer"
    index_path = viewer_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"viewer index not found: {index_path}")

    handler = type("BoundViewerHandler", (ViewerHandler,), {})
    handler.output_dir = output_dir
    handler.viewer_dir = viewer_dir
    handler.index_data = json.loads(index_path.read_text(encoding="utf-8"))
    handler.volume_cache = OrderedDict()
    handler.render_cache = OrderedDict()
    handler.volume_cache_lock = threading.RLock()
    handler.render_cache_lock = threading.RLock()
    handler.max_cached_volumes = 4
    handler.max_cached_render_volumes = 2

    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://localhost:{port}"
    print("")
    print("Viewer running.")
    print("")
    print("If this viewer is running on your own computer, open:")
    print(f"  {url}")
    print("")
    print("If this viewer is running on cluster, open a NEW terminal on your own computer")
    print("and start this SSH tunnel:")
    print(f"  ssh -L {port}:<node-ip>:{port} <user>@<login-host>")
    print("")
    print("Then open this same link in your browser:")
    print(f"  {url}")
    print("")
    print("Keep this terminal open. Press Ctrl+C to stop the viewer.")
    print("If the viewer is running on a different cluster node, replace <node-ip> with that node IP.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the HCAT analysis viewer.")
    parser.add_argument("output_dir", type=Path, help="Analysis output directory.")
    parser.add_argument("--host", default=os.environ.get("HCAT_VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HCAT_VIEWER_PORT", "8765")))
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    args = parser.parse_args()
    serve(args.output_dir, host=args.host, port=args.port, open_browser=args.open)


INDEX_HTML = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HCAT Analysis Viewer</title>
  <link rel="stylesheet" href="/assets/viewer.css">
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>HCAT Analysis Viewer</h1>
        <p>Seleziona caso e ostio; i volumi vengono letti on demand dal server locale.</p>
      </div>
      <label class="toggle"><input id="outliersOnly" type="checkbox"> solo outlier</label>
    </header>
    <section class="controls">
      <label>Caso <select id="caseSelect"></select></label>
      <label>Ostio <select id="sideSelect"></select></label>
      <label>Soglia heatmap 3D <input id="threshold" type="range" min="0.05" max="0.9" step="0.05" value="0.2"></label>
      <div id="metrics" class="metrics"></div>
    </section>
    <section class="views">
      <div class="panel">
        <div class="view-title"><span>Assiale Z</span><span id="zValue"></span></div>
        <input id="zSlider" type="range" min="0" value="0">
        <canvas id="axial"></canvas>
      </div>
      <div class="panel">
        <div class="view-title"><span>Coronale Y</span><span id="yValue"></span></div>
        <input id="ySlider" type="range" min="0" value="0">
        <canvas id="coronal"></canvas>
      </div>
      <div class="panel">
        <div class="view-title"><span>Sagittale X</span><span id="xValue"></span></div>
        <input id="xSlider" type="range" min="0" value="0">
        <canvas id="sagittal"></canvas>
      </div>
    </section>
    <section class="panel">
      <div class="view-title"><span>Spazio 3D</span><span>verde: label, rosso: prediction, arancio: heatmap selezionata</span></div>
      <div id="plot3d"></div>
    </section>
  </main>
  <script src="/assets/viewer.js"></script>
</body>
</html>
"""


VIEWER_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #151922;
  --muted: #647084;
  --border: #d9dee7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101318;
    --panel: #171b22;
    --text: #eef2f7;
    --muted: #aab3c2;
    --border: #303744;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1240px;
  margin: 0 auto;
  padding: 20px;
}
header {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 16px;
  margin-bottom: 16px;
}
h1 { font-size: 22px; margin: 0 0 6px; }
p { margin: 0; color: var(--muted); }
.toggle { color: var(--muted); font-size: 14px; }
.controls, .panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.controls {
  display: grid;
  grid-template-columns: minmax(260px, 1.3fr) 110px minmax(180px, 0.8fr) minmax(260px, 1.3fr);
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
}
label {
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
}
select, input[type="range"] { width: 100%; }
select {
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
}
.metrics {
  color: var(--text);
  font-size: 13px;
  line-height: 1.4;
}
.views {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-bottom: 16px;
}
.view-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}
canvas {
  display: block;
  width: 100%;
  background: #000;
  border-radius: 6px;
}
#plot3d { height: 560px; }
@media (max-width: 960px) {
  .controls, .views { grid-template-columns: 1fr; }
  header { display: block; }
}
"""


VIEWER_JS = """
let indexData = null;
let visibleCases = [];
const sliceResponseCache = new Map();
const maxBrowserCachedSlices = 900;

const els = {
  caseSelect: document.getElementById('caseSelect'),
  sideSelect: document.getElementById('sideSelect'),
  outliersOnly: document.getElementById('outliersOnly'),
  threshold: document.getElementById('threshold'),
  metrics: document.getElementById('metrics'),
  zSlider: document.getElementById('zSlider'),
  ySlider: document.getElementById('ySlider'),
  xSlider: document.getElementById('xSlider'),
  zValue: document.getElementById('zValue'),
  yValue: document.getElementById('yValue'),
  xValue: document.getElementById('xValue'),
  axial: document.getElementById('axial'),
  coronal: document.getElementById('coronal'),
  sagittal: document.getElementById('sagittal'),
};

function currentCaseIndex() {
  return Number(els.caseSelect.value || 0);
}

function currentCase() {
  return visibleCases[currentCaseIndex()];
}

function currentRecord() {
  const side = els.sideSelect.value;
  const c = currentCase();
  return c.records.find(r => r.side === side) || c.records[0];
}

function recordPoint(record, kind) {
  return record ? record[kind] : null;
}

function pointVisible(point, plane, sliceIndex) {
  if (!point) return false;
  const axis = plane === 'z' ? 0 : plane === 'y' ? 1 : 2;
  return Math.abs(point[axis] - sliceIndex) <= 1;
}

function drawPoint(ctx, px, py, color, label) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px - 7, py);
  ctx.lineTo(px + 7, py);
  ctx.moveTo(px, py - 7);
  ctx.lineTo(px, py + 7);
  ctx.stroke();
  ctx.font = '13px system-ui';
  ctx.fillText(label, px + 8, py - 8);
  ctx.restore();
}

function decodeBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

async function fetchSlice(source, plane, sliceIndex) {
  const cacheKey = [
    currentCase()?.case_id || currentCaseIndex(),
    els.sideSelect.value,
    source,
    plane,
    sliceIndex,
  ].join('|');
  if (sliceResponseCache.has(cacheKey)) {
    const cached = sliceResponseCache.get(cacheKey);
    sliceResponseCache.delete(cacheKey);
    sliceResponseCache.set(cacheKey, cached);
    return cached;
  }

  const params = new URLSearchParams({
    case: String(currentCaseIndex()),
    side: els.sideSelect.value,
    source,
    plane,
    index: String(sliceIndex),
  });
  const response = await fetch(`/api/slice_raw?${params.toString()}`);
  if (!response.ok) {
    let message = 'slice error';
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch (_err) {}
    throw new Error(message);
  }
  const payload = {
    width: Number(response.headers.get('X-Slice-Width')),
    height: Number(response.headers.get('X-Slice-Height')),
    shape: [
      Number(response.headers.get('X-Shape-Z')),
      Number(response.headers.get('X-Shape-Y')),
      Number(response.headers.get('X-Shape-X')),
    ],
    spacing_zyx: [
      Number(response.headers.get('X-Spacing-Z')),
      Number(response.headers.get('X-Spacing-Y')),
      Number(response.headers.get('X-Spacing-X')),
    ],
    data: new Uint8Array(await response.arrayBuffer()),
  };
  sliceResponseCache.set(cacheKey, payload);
  while (sliceResponseCache.size > maxBrowserCachedSlices) {
    const firstKey = sliceResponseCache.keys().next().value;
    sliceResponseCache.delete(firstKey);
  }
  return payload;
}

function pointToCanvas(point, plane) {
  if (plane === 'z') return [point[2], point[1]];
  if (plane === 'y') return [point[2], point[0]];
  return [point[1], point[0]];
}

async function drawSlice(canvas, plane, slider, valueEl) {
  const sliceIndex = Number(slider.value);
  valueEl.textContent = String(sliceIndex);
  let image, heatmap = null;
  try {
    image = await fetchSlice('image', plane, sliceIndex);
  } catch (_err) {
    image = await fetchSlice('heatmap', plane, sliceIndex);
  }
  try {
    heatmap = await fetchSlice('heatmap', plane, sliceIndex);
  } catch (_err) {
    heatmap = null;
  }

  canvas.width = image.width;
  canvas.height = image.height;
  const spacing = image.spacing_zyx || [1, 1, 1];
  if (plane === 'z') canvas.style.aspectRatio = `${image.width * spacing[2]} / ${image.height * spacing[1]}`;
  if (plane === 'y') canvas.style.aspectRatio = `${image.width * spacing[2]} / ${image.height * spacing[0]}`;
  if (plane === 'x') canvas.style.aspectRatio = `${image.width * spacing[1]} / ${image.height * spacing[0]}`;

  const ctx = canvas.getContext('2d');
  const imageBytes = image.data;
  const heatBytes = heatmap ? heatmap.data : null;
  const img = ctx.createImageData(image.width, image.height);
  for (let i = 0; i < imageBytes.length; i++) {
    const v = imageBytes[i];
    const h = heatBytes ? heatBytes[i] / 255 : 0;
    const alpha = Math.min(0.72, h * 0.72);
    const off = i * 4;
    img.data[off] = Math.round(v * (1 - alpha) + 245 * alpha);
    img.data[off + 1] = Math.round(v * (1 - alpha) + 142 * alpha);
    img.data[off + 2] = Math.round(v * (1 - alpha) + 20 * alpha);
    img.data[off + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  const rec = currentRecord();
  const label = recordPoint(rec, 'label');
  const pred = recordPoint(rec, 'prediction');
  if (pointVisible(label, plane, sliceIndex)) {
    const [x, y] = pointToCanvas(label, plane);
    drawPoint(ctx, x, y, '#18a558', 'GT');
  }
  if (pointVisible(pred, plane, sliceIndex)) {
    const [x, y] = pointToCanvas(pred, plane);
    drawPoint(ctx, x, y, '#dc2626', 'P');
  }
}

function setSliders() {
  const c = currentCase();
  const rec = currentRecord();
  const shape = c.shape && c.shape.every(v => v !== null) ? c.shape : [64, 64, 64];
  els.zSlider.max = Math.max(0, shape[0] - 1);
  els.ySlider.max = Math.max(0, shape[1] - 1);
  els.xSlider.max = Math.max(0, shape[2] - 1);
  const p = rec.prediction || rec.label || [shape[0] / 2, shape[1] / 2, shape[2] / 2];
  els.zSlider.value = Math.round(Math.min(Math.max(p[0], 0), shape[0] - 1));
  els.ySlider.value = Math.round(Math.min(Math.max(p[1], 0), shape[1] - 1));
  els.xSlider.value = Math.round(Math.min(Math.max(p[2], 0), shape[2] - 1));
}

async function drawAllSlices() {
  await Promise.all([
    drawSlice(els.axial, 'z', els.zSlider, els.zValue),
    drawSlice(els.coronal, 'y', els.ySlider, els.yValue),
    drawSlice(els.sagittal, 'x', els.xSlider, els.xValue),
  ]).catch(err => { els.metrics.innerHTML += `<br><span class="error">${err.message}</span>`; });
}

function validShape(shape) {
  return Array.isArray(shape) && shape.length >= 3 && shape.every(v => Number.isFinite(Number(v)) && Number(v) > 0);
}

function validSpacing(spacing) {
  return Array.isArray(spacing) && spacing.length >= 3 && spacing.every(v => Number.isFinite(Number(v)) && Number(v) > 0);
}

function caseShape(c) {
  return validShape(c.shape) ? c.shape.map(Number) : [64, 64, 64];
}

function caseSpacing(c) {
  return validSpacing(c.spacing_zyx) ? c.spacing_zyx.map(Number) : [1, 1, 1];
}

function zyxToXyzMm(point, spacing) {
  return [point[2] * spacing[2], point[1] * spacing[1], point[0] * spacing[0]];
}

function shapeToXyzMm(shape, spacing) {
  return [shape[2] * spacing[2], shape[1] * spacing[1], shape[0] * spacing[0]];
}

function pointInsideShape(point, shape) {
  return point &&
    point[0] >= 0 && point[0] <= shape[0] - 1 &&
    point[1] >= 0 && point[1] <= shape[1] - 1 &&
    point[2] >= 0 && point[2] <= shape[2] - 1;
}

function volumeBoxTrace(shape, spacing) {
  const [xMax, yMax, zMax] = shapeToXyzMm(shape, spacing);
  const x = [0, xMax, xMax, 0, 0, null, 0, xMax, xMax, 0, 0, null, 0, 0, null, xMax, xMax, null, xMax, xMax, null, 0, 0];
  const y = [0, 0, yMax, yMax, 0, null, 0, 0, yMax, yMax, 0, null, 0, 0, null, 0, 0, null, yMax, yMax, null, yMax, yMax];
  const z = [0, 0, 0, 0, 0, null, zMax, zMax, zMax, zMax, zMax, null, 0, zMax, null, 0, zMax, null, 0, zMax, null, 0, zMax];
  return {
    type: 'scatter3d',
    mode: 'lines',
    x, y, z,
    line: {color: 'rgba(148, 163, 184, 0.55)', width: 2},
    name: 'volume bounds',
    hoverinfo: 'skip',
  };
}

function markerTrace(point, spacing, shape, label, color, size, symbol) {
  const [x, y, z] = zyxToXyzMm(point, spacing);
  const insideText = pointInsideShape(point, shape) ? 'inside volume' : 'OUTSIDE volume';
  return {
    type: 'scatter3d',
    mode: 'markers+text',
    x: [x], y: [y], z: [z],
    text: [label],
    textposition: 'top center',
    marker: {
      size,
      color,
      symbol,
      line: {color: '#ffffff', width: 1},
    },
    name: label,
    hovertemplate:
      `${label}<br>` +
      `${insideText}<br>` +
      `x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>z=%{z:.1f} mm<br>` +
      `voxel z=${point[0].toFixed(1)}, y=${point[1].toFixed(1)}, x=${point[2].toFixed(1)}` +
      `<extra></extra>`,
  };
}

function paddedRange(values, fallbackMax) {
  const finite = values.filter(v => Number.isFinite(v));
  const lo = Math.min(0, ...finite);
  const hi = Math.max(fallbackMax, ...finite);
  const pad = Math.max((hi - lo) * 0.05, 1);
  return [lo - pad, hi + pad];
}

function outsideWarnings(c) {
  const shape = caseShape(c);
  const warnings = [];
  for (const r of c.records) {
    if (!r.label) warnings.push(`label ${r.side || '-'} missing`);
    if (r.label && !pointInsideShape(r.label, shape)) warnings.push(`label ${r.side || '-'} outside volume`);
    if (r.prediction && !pointInsideShape(r.prediction, shape)) warnings.push(`prediction ${r.side || '-'} outside volume`);
  }
  return warnings;
}

async function draw3d() {
  const c = currentCase();
  const rec = currentRecord();
  const traces = [];
  const shape = caseShape(c);
  const spacing = caseSpacing(c);
  const [xMax, yMax, zMax] = shapeToXyzMm(shape, spacing);
  traces.push(volumeBoxTrace(shape, spacing));
  const params = new URLSearchParams({
    case: String(currentCaseIndex()),
    side: els.sideSelect.value,
    threshold: els.threshold.value,
  });
  try {
    const response = await fetch(`/api/heatmap3d?${params.toString()}`);
    const payload = await response.json();
    if (payload.points && payload.points.length) {
      traces.push({
        type: 'scatter3d',
        mode: 'markers',
        x: payload.points.map(p => p[0] * spacing[2]),
        y: payload.points.map(p => p[1] * spacing[1]),
        z: payload.points.map(p => p[2] * spacing[0]),
        marker: {
          size: 3,
          color: payload.points.map(p => p[3]),
          colorscale: 'YlOrRd',
          opacity: 0.28,
          showscale: false,
        },
        name: 'heatmap'
      });
    }
  } catch (_err) {}

  for (const r of c.records) {
    if (r.label) {
      traces.push(markerTrace(
        r.label,
        spacing,
        shape,
        `label ${r.side || '-'}`,
        '#18a558',
        r.side === rec.side ? 7 : 5,
        'diamond'
      ));
    }
    if (r.prediction) {
      traces.push(markerTrace(
        r.prediction,
        spacing,
        shape,
        `prediction ${r.side || '-'}`,
        '#dc2626',
        r.side === rec.side ? 8 : 5,
        'circle'
      ));
    }
  }

  const allX = [];
  const allY = [];
  const allZ = [];
  for (const r of c.records) {
    for (const p of [r.label, r.prediction]) {
      if (p) {
        const [x, y, z] = zyxToXyzMm(p, spacing);
        allX.push(x); allY.push(y); allZ.push(z);
      }
    }
  }
  allX.push(0, xMax);
  allY.push(0, yMax);
  allZ.push(0, zMax);
  const aspectBase = Math.max(xMax, yMax, zMax, 1);
  Plotly.react('plot3d', traces, {
    margin: {l: 0, r: 0, b: 0, t: 0},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      xaxis: {title: 'x mm', range: paddedRange(allX, xMax), zeroline: false},
      yaxis: {title: 'y mm', range: paddedRange(allY, yMax), zeroline: false},
      zaxis: {title: 'z mm', range: paddedRange(allZ, zMax), zeroline: false},
      aspectmode: 'manual',
      aspectratio: {
        x: Math.max(xMax / aspectBase, 0.05),
        y: Math.max(yMax / aspectBase, 0.05),
        z: Math.max(zMax / aspectBase, 0.05),
      },
      camera: {eye: {x: 1.55, y: 1.65, z: 1.15}},
    },
    showlegend: true,
  }, {responsive: true, displayModeBar: true});
}

function formatMetric(value, suffix = '') {
  return value === null || value === undefined ? '-' : `${Number(value).toFixed(3)}${suffix}`;
}

function renderMetrics() {
  const c = currentCase();
  const r = currentRecord();
  const warnings = outsideWarnings(c);
  els.metrics.innerHTML = `
    <strong>${c.case_id}</strong> · lato <strong>${r.side || '-'}</strong>
    · errore <strong>${formatMetric(r.distance_to_label_mm, ' mm')}</strong><br>
    entropy norm ${formatMetric(r.heatmap_entropy_norm)}
    · spread ${formatMetric(r.heatmap_spread_mm, ' mm')}
    · prominence p95 ${formatMetric(r.heatmap_peak_prominence_p95)}
    ${warnings.length ? `<br><span class="error">${warnings.join(' · ')}</span>` : ''}
  `;
}

function refreshSideOptions() {
  const c = currentCase();
  const selected = els.sideSelect.value;
  els.sideSelect.innerHTML = '';
  for (const r of c.records) {
    const opt = document.createElement('option');
    opt.value = r.side || '';
    opt.textContent = r.side || '-';
    els.sideSelect.appendChild(opt);
  }
  if ([...els.sideSelect.options].some(o => o.value === selected)) els.sideSelect.value = selected;
}

function refreshCases() {
  const old = currentCase()?.case_id;
  visibleCases = indexData.cases.filter(c => !els.outliersOnly.checked || c.records.some(r => r.is_outlier));
  els.caseSelect.innerHTML = '';
  visibleCases.forEach((c, i) => {
    const maxErr = Math.max(...c.records.map(r => r.distance_to_label_mm || 0));
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `${c.case_id} · max ${maxErr.toFixed(2)} mm`;
    els.caseSelect.appendChild(opt);
  });
  const oldIndex = visibleCases.findIndex(c => c.case_id === old);
  if (oldIndex >= 0) els.caseSelect.value = String(oldIndex);
  refreshSideOptions();
}

function debounce(fn, waitMs) {
  let timeoutId = null;
  return (...args) => {
    if (timeoutId !== null) window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => {
      timeoutId = null;
      fn(...args);
    }, waitMs);
  };
}

async function renderAll(resetSliders = false) {
  if (!visibleCases.length) {
    els.metrics.textContent = 'Nessun caso disponibile.';
    return;
  }
  refreshSideOptions();
  if (resetSliders) setSliders();
  renderMetrics();
  await drawAllSlices();
  await draw3d();
}

fetch('/api/index')
  .then(r => r.json())
  .then(data => {
    indexData = data;
    refreshCases();
  renderAll(true);
});

const drawAxialDebounced = debounce(() => drawSlice(els.axial, 'z', els.zSlider, els.zValue), 80);
const drawCoronalDebounced = debounce(() => drawSlice(els.coronal, 'y', els.ySlider, els.yValue), 80);
const drawSagittalDebounced = debounce(() => drawSlice(els.sagittal, 'x', els.xSlider, els.xValue), 80);
const draw3dDebounced = debounce(draw3d, 180);

els.caseSelect.addEventListener('change', () => renderAll(true));
els.sideSelect.addEventListener('change', () => renderAll(true));
els.outliersOnly.addEventListener('change', () => { refreshCases(); renderAll(true); });
els.threshold.addEventListener('input', draw3dDebounced);
els.zSlider.addEventListener('input', drawAxialDebounced);
els.ySlider.addEventListener('input', drawCoronalDebounced);
els.xSlider.addEventListener('input', drawSagittalDebounced);
"""


if __name__ == "__main__":
    main()
