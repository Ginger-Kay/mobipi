#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ORDER = {"CloseDrawer": ["E", "D", "A", "A0"], "CloseSingleDoor": ["A0", "A", "D", "E"]}


def labels(root: Path, task: str) -> dict[str, dict]:
    rows = {}
    for route in ORDER[task]:
        path = root / "workers" / task / "route-records" / f"{route}.json"
        rows[route] = json.loads(path.read_text())
    return rows


def frames(path: str) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame)[:1080, :1920, :3] for frame in reader]
    finally:
        reader.close()


def annotate(frame: np.ndarray, label: str, row: dict) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    metrics = row.get("runtime_metrics", {})
    text = f"{label} | base net {float(metrics.get('actual_base_net_m', 0.0)):.3f} m | progress {float(row.get('task_progress_after', 0.0)):.3f} | collision {row.get('collision', False)}"
    draw.rectangle((0, 0, 1920, 42), fill=(0, 0, 0))
    draw.text((18, 10), text, fill=(255, 255, 255))
    return np.asarray(image)


def compose(root: Path, task: str) -> dict:
    rows = labels(root, task)
    streams = {}
    for route in ORDER[task]:
        row = rows[route]
        stream = frames(row["video_path"])
        if not stream:
            raise RuntimeError(f"empty route video: {task}/{route}")
        streams[route] = [annotate(frame, route, row) for frame in stream]
    length = max(len(value) for value in streams.values())
    out = root / "videos"
    out.mkdir(parents=True, exist_ok=True)
    master = out / f"{task}-E-D-A-A0-master-3840x2160.mp4"
    share = out / f"{task}-E-D-A-A0-share-1920x1080.mp4"
    with imageio.get_writer(master, fps=20, codec="libx264", macro_block_size=1, quality=8) as writer_master, imageio.get_writer(share, fps=20, codec="libx264", macro_block_size=1, quality=8) as writer_share:
        for index in range(length):
            panels = []
            for route in ORDER[task]:
                panel = streams[route][min(index, len(streams[route]) - 1)]
                panels.append(panel)
            top = np.concatenate([panels[0], panels[1]], axis=1)
            bottom = np.concatenate([panels[2], panels[3]], axis=1)
            master_frame = np.concatenate([top, bottom], axis=0)
            writer_master.append_data(master_frame)
            writer_share.append_data(np.asarray(Image.fromarray(master_frame).resize((1920, 1080), Image.Resampling.LANCZOS)))
    return {"task": task, "master_path": str(master), "share_path": str(share), "master_shape": [3840, 2160], "share_shape": [1920, 1080], "frames": length, "route_order": ORDER[task]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    receipts = [compose(args.artifact_root, task) for task in ORDER]
    (args.artifact_root / "video-compositor-receipt.json").write_text(json.dumps({"schema_version": "v1-video-compositor-v1.0.1", "receipts": receipts}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
