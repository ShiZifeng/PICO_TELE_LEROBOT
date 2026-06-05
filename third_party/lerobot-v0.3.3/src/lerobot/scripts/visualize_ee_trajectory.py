#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


LEFT_SLICE = slice(0, 3)
RIGHT_SLICE = slice(7, 10)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, max_points).round().astype(np.int64)


def load_episode(path: Path, max_points: int) -> dict:
    table = pq.read_table(path, columns=["timestamp", "action", "observation.state", "episode_index"])
    data = table.to_pydict()
    episode_index = int(data["episode_index"][0])
    timestamp = np.asarray(data["timestamp"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    state = np.asarray(data["observation.state"], dtype=np.float32)
    indices = downsample_indices(len(timestamp), max_points)

    return {
        "episode_index": episode_index,
        "length": int(len(timestamp)),
        "timestamp": timestamp[indices].round(4).tolist(),
        "state_left": state[indices, LEFT_SLICE].round(6).tolist(),
        "state_right": state[indices, RIGHT_SLICE].round(6).tolist(),
        "action_left": action[indices, LEFT_SLICE].round(6).tolist(),
        "action_right": action[indices, RIGHT_SLICE].round(6).tolist(),
    }


def dataset_bounds(episodes: list[dict]) -> dict:
    arrays = []
    for episode in episodes:
        for key in ["state_left", "state_right", "action_left", "action_right"]:
            arrays.append(np.asarray(episode[key], dtype=np.float32))
    points = np.concatenate(arrays, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = float(max(maxs - mins))
    if span <= 0:
        span = 1.0
    return {
        "min": mins.round(6).tolist(),
        "max": maxs.round(6).tolist(),
        "center": center.round(6).tolist(),
        "span": span,
    }


def html_escape_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def build_html(payload: dict) -> str:
    data = html_escape_json(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EE Trajectory Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1e2329;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px;
      border-bottom: 1px solid #d9dde3;
      background: #fff;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .meta {{
      color: #66707d;
      font-size: 13px;
      margin-top: 4px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 13px;
      color: #36404c;
      white-space: nowrap;
    }}
    select, input[type="range"] {{
      accent-color: #2563eb;
    }}
    select {{
      min-width: 108px;
      border: 1px solid #c9d0d9;
      border-radius: 6px;
      background: #fff;
      padding: 7px 9px;
      font-size: 13px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 78px);
    }}
    .panel {{
      min-width: 0;
      background: #fff;
      border: 1px solid #d9dde3;
      border-radius: 8px;
      overflow: hidden;
    }}
    .panel-title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 11px 14px;
      border-bottom: 1px solid #e2e6eb;
      font-size: 13px;
      color: #4d5865;
    }}
    .canvas-wrap {{
      position: relative;
      min-height: 500px;
      height: calc(100vh - 154px);
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      background: #fbfcfd;
    }}
    .legend {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px 12px;
      padding: 12px 14px;
      font-size: 13px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .swatch {{
      width: 22px;
      height: 3px;
      border-radius: 2px;
      flex: none;
    }}
    .stats {{
      padding: 0 14px 14px;
      font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #36404c;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    .hint {{
      padding: 12px 14px;
      border-top: 1px solid #e2e6eb;
      color: #66707d;
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 980px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .controls {{ justify-content: flex-start; }}
      main {{ grid-template-columns: 1fr; }}
      .canvas-wrap {{ height: 62vh; min-height: 390px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>EE Trajectory Viewer</h1>
      <div class="meta" id="datasetMeta"></div>
    </div>
    <div class="controls">
      <label>Episode <select id="episodeSelect"></select></label>
      <label>Source <select id="sourceSelect">
        <option value="state">observation.state</option>
        <option value="action">action</option>
        <option value="both">state + action</option>
      </select></label>
      <label>Frame <input id="frameSlider" type="range" min="0" max="0" value="0"></label>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="panel-title"><span>3D EE trajectory</span><span id="frameLabel"></span></div>
      <div class="canvas-wrap"><canvas id="sceneCanvas"></canvas></div>
    </section>
    <section class="panel">
      <div class="panel-title"><span>Details</span><span>meters</span></div>
      <div class="legend">
        <div class="legend-item"><span class="swatch" style="background:#1f77b4"></span>state left</div>
        <div class="legend-item"><span class="swatch" style="background:#d62728"></span>state right</div>
        <div class="legend-item"><span class="swatch" style="background:#17becf"></span>action left</div>
        <div class="legend-item"><span class="swatch" style="background:#ff7f0e"></span>action right</div>
      </div>
      <div class="stats" id="stats"></div>
      <div class="hint">Drag to rotate. Use the mouse wheel or trackpad scroll to zoom. The gray box is the global dataset bounds and the colored dots mark the selected frame.</div>
    </section>
  </main>
  <script>
    const payload = {data};
    const colors = {{
      state_left: "#1f77b4",
      state_right: "#d62728",
      action_left: "#17becf",
      action_right: "#ff7f0e",
    }};
    const labels = {{
      state_left: "state left",
      state_right: "state right",
      action_left: "action left",
      action_right: "action right",
    }};
    const select = document.getElementById("episodeSelect");
    const sourceSelect = document.getElementById("sourceSelect");
    const slider = document.getElementById("frameSlider");
    const frameLabel = document.getElementById("frameLabel");
    const stats = document.getElementById("stats");
    const canvas = document.getElementById("sceneCanvas");
    const camera = {{ yaw: -0.72, pitch: 0.42, zoom: 1.0, dragging: false, lastX: 0, lastY: 0 }};

    document.getElementById("datasetMeta").textContent =
      `${{payload.dataset_root}}  |  ${{payload.episodes.length}} episodes  |  bounds x/y/z ${{payload.bounds.min.map(v => v.toFixed(3)).join(", ")}} to ${{payload.bounds.max.map(v => v.toFixed(3)).join(", ")}}`;

    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = `All episodes (${{payload.episodes.length}})`;
    select.appendChild(allOption);

    for (const ep of payload.episodes) {{
      const option = document.createElement("option");
      option.value = ep.episode_index;
      option.textContent = `ep${{String(ep.episode_index).padStart(3, "0")}} (${{ep.length}} frames)`;
      select.appendChild(option);
    }}

    function isAllEpisodes() {{
      return select.value === "all";
    }}

    function currentEpisode() {{
      const epIndex = Number(select.value);
      return payload.episodes.find(ep => ep.episode_index === epIndex) || payload.episodes[0];
    }}

    function activeKeys() {{
      const source = sourceSelect.value;
      if (source === "state") return ["state_left", "state_right"];
      if (source === "action") return ["action_left", "action_right"];
      return ["state_left", "state_right", "action_left", "action_right"];
    }}

    function resizeCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.floor(rect.width * scale));
      const height = Math.max(240, Math.floor(rect.height * scale));
      if (canvas.width !== width || canvas.height !== height) {{
        canvas.width = width;
        canvas.height = height;
      }}
      return {{ width, height, scale }};
    }}

    function rotatePoint(point) {{
      const center = payload.bounds.center;
      let x = point[0] - center[0];
      let y = point[1] - center[1];
      let z = point[2] - center[2];

      const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
      const x1 = cy * x + sy * y;
      const y1 = -sy * x + cy * y;

      const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
      const y2 = cp * y1 - sp * z;
      const z2 = sp * y1 + cp * z;
      return [x1, y2, z2];
    }}

    function project3d(point, width, height) {{
      const p = rotatePoint(point);
      const span = Math.max(payload.bounds.span, 0.001);
      const focal = 3.2 * span;
      const depth = focal - p[2];
      const scale = Math.min(width, height) * 0.72 * camera.zoom / span;
      const perspective = focal / Math.max(0.15 * span, depth);
      return [
        width / 2 + p[0] * scale * perspective,
        height / 2 - p[1] * scale * perspective,
        p[2],
        perspective,
      ];
    }}

    function drawSegment(ctx, a, b, width, height, color, alpha, lineWidth) {{
      const pa = project3d(a, width, height);
      const pb = project3d(b, width, height);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }}

    function drawBounds(ctx, width, height) {{
      const min = payload.bounds.min;
      const max = payload.bounds.max;
      const corners = [
        [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]],
        [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]],
      ];
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      ctx.strokeStyle = "#c8d0da";
      ctx.lineWidth = 1;
      for (const [a, b] of edges) drawSegment(ctx, corners[a], corners[b], width, height, "#c8d0da", 0.72, 1);

      const axes = [
        {{ from: payload.bounds.center, to: [max[0], payload.bounds.center[1], payload.bounds.center[2]], color: "#44515f", label: "x" }},
        {{ from: payload.bounds.center, to: [payload.bounds.center[0], max[1], payload.bounds.center[2]], color: "#44515f", label: "y" }},
        {{ from: payload.bounds.center, to: [payload.bounds.center[0], payload.bounds.center[1], max[2]], color: "#44515f", label: "z" }},
      ];
      ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
      ctx.fillStyle = "#44515f";
      for (const axis of axes) {{
        drawSegment(ctx, axis.from, axis.to, width, height, axis.color, 0.9, 1.4);
        const labelPoint = project3d(axis.to, width, height);
        ctx.fillText(axis.label, labelPoint[0] + 5, labelPoint[1] - 5);
      }}
    }}

    function drawTrajectory(ctx, ep, key, width, height, withMarker, frame) {{
      const points = ep[key];
      const projected = points.map(p => project3d(p, width, height));
      ctx.strokeStyle = colors[key];
      ctx.lineWidth = withMarker ? (key.startsWith("action") ? 1.5 : 2.2) : 0.95;
      ctx.globalAlpha = withMarker ? (key.startsWith("action") ? 0.64 : 0.88) : 0.22;
      ctx.beginPath();
      for (let i = 0; i < projected.length; i++) {{
        const [x, y] = projected[i];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (!withMarker) return;
      const marker = projected[Math.min(frame, projected.length - 1)];
      ctx.fillStyle = colors[key];
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(marker[0], marker[1], 5.5 + marker[3] * 1.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }}

    function drawScene() {{
      const allMode = isAllEpisodes();
      const ep = currentEpisode();
      const frame = Number(slider.value);
      const {{ width, height }} = resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(0, 0, width, height);
      drawBounds(ctx, width, height);

      if (allMode) {{
        for (const episode of payload.episodes) {{
          for (const key of activeKeys()) {{
            drawTrajectory(ctx, episode, key, width, height, false, 0);
          }}
        }}
      }} else {{
        for (const key of activeKeys()) {{
          drawTrajectory(ctx, ep, key, width, height, true, frame);
        }}
      }}
    }}

    function render() {{
      const allMode = isAllEpisodes();
      const ep = currentEpisode();
      const max = Math.max(0, ep.timestamp.length - 1);
      slider.disabled = allMode;
      if (Number(slider.max) !== max) {{
        slider.max = String(max);
        slider.value = "0";
      }}
      const frame = Number(slider.value);
      const time = ep.timestamp[Math.min(frame, ep.timestamp.length - 1)] || 0;
      frameLabel.textContent = allMode
        ? `${{payload.episodes.length}} episodes overlaid`
        : `sample ${{frame}} / ${{max}}  |  t=${{Number(time).toFixed(3)}}s`;
      drawScene();

      const totalFrames = payload.episodes.reduce((sum, item) => sum + item.length, 0);
      const totalPoints = payload.episodes.reduce((sum, item) => sum + item.timestamp.length, 0);
      const lines = allMode
        ? [`episodes: ${{payload.episodes.length}}`, `frames: ${{totalFrames}}`, `displayed points per source: ${{totalPoints}}`]
        : [`episode: ${{ep.episode_index}}`, `frames: ${{ep.length}}`, `displayed points: ${{ep.timestamp.length}}`];
      lines.push(`view: yaw=${{camera.yaw.toFixed(2)}} pitch=${{camera.pitch.toFixed(2)}} zoom=${{camera.zoom.toFixed(2)}}`);
      if (!allMode) {{
        for (const key of activeKeys()) {{
          const p = ep[key][Math.min(frame, ep[key].length - 1)];
          lines.push(`${{labels[key]}}: x=${{p[0].toFixed(4)}} y=${{p[1].toFixed(4)}} z=${{p[2].toFixed(4)}}`);
        }}
      }}
      stats.textContent = lines.join("\\n");
    }}

    select.addEventListener("change", render);
    sourceSelect.addEventListener("change", render);
    slider.addEventListener("input", render);
    canvas.addEventListener("pointerdown", event => {{
      camera.dragging = true;
      camera.lastX = event.clientX;
      camera.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener("pointermove", event => {{
      if (!camera.dragging) return;
      const dx = event.clientX - camera.lastX;
      const dy = event.clientY - camera.lastY;
      camera.lastX = event.clientX;
      camera.lastY = event.clientY;
      camera.yaw += dx * 0.008;
      camera.pitch = Math.max(-1.35, Math.min(1.35, camera.pitch + dy * 0.008));
      render();
    }});
    canvas.addEventListener("pointerup", event => {{
      camera.dragging = false;
      canvas.releasePointerCapture(event.pointerId);
    }});
    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      camera.zoom = Math.max(0.35, Math.min(4.5, camera.zoom * Math.exp(-event.deltaY * 0.001)));
      render();
    }}, {{ passive: false }});
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""


def create_viewer(dataset_root: Path, output_dir: Path, episodes: list[int] | None, max_points: int) -> Path:
    data_dir = dataset_root / "data/chunk-000"
    info_path = dataset_root / "meta/info.json"
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find parquet directory: {data_dir}")
    if not info_path.is_file():
        raise FileNotFoundError(f"Could not find dataset info: {info_path}")

    info = json.loads(info_path.read_text())
    names = info["features"]["action"].get("names") or []
    if len(names) != 14 or "left_ee.x" not in names or "right_ee.x" not in names:
        raise ValueError(
            "This viewer expects the 14-D EE dataset produced by convert_dataset_to_ee.py."
        )

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if episodes is not None:
        episode_set = set(episodes)
        parquet_files = [p for p in parquet_files if int(p.stem.split("_")[-1]) in episode_set]
    if not parquet_files:
        raise ValueError("No episodes selected.")

    loaded = [load_episode(path, max_points) for path in parquet_files]
    payload = {
        "dataset_root": str(dataset_root),
        "features": names,
        "bounds": dataset_bounds(loaded),
        "episodes": loaded,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_root.name}_ee_trajectory.html"
    output_path.write_text(build_html(payload))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local HTML viewer for 14-D EE trajectory datasets.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/lift_basket2_ee"))
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ee_trajectory"))
    parser.add_argument("--max-points", type=int, default=1200)
    args = parser.parse_args()

    output_path = create_viewer(args.dataset_root, args.output_dir, args.episodes, args.max_points)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
