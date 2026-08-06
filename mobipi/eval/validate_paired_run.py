"""Validate a completed paired-rollout directory without loading the policy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
import numpy as np

from mobipi.utils.door_diagnostics import validate_diagnostic_trajectory
from mobipi.utils.paired_rollout_utils import sha256_file, validate_manifest


@click.command()
@click.option("--run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--expected_candidates", required=True, type=int)
def main(run_dir, expected_candidates):
    run_dir = Path(run_dir).resolve()
    batch_path = run_dir / "batch_manifest.json"
    batch = json.loads(batch_path.read_text())
    results = batch.get("results", [])
    if len(results) != expected_candidates:
        raise click.ClickException(
            f"expected {expected_candidates} candidates, found {len(results)}"
        )
    seen = set()
    for result in results:
        manifest_path = Path(result["manifest_uri"]).resolve()
        if run_dir not in manifest_path.parents:
            raise click.ClickException(f"manifest escapes run directory: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest)
        if manifest["run_id"] in seen:
            raise click.ClickException(f"duplicate run_id: {manifest['run_id']}")
        seen.add(manifest["run_id"])
        if manifest["status"] != "completed" or manifest["execution"]["exit_code"] != 0:
            raise click.ClickException(
                f"candidate is not complete: {manifest['run_id']} status={manifest['status']}"
            )
        formats = set()
        artifact_by_name = {}
        for artifact in manifest["artifacts"]:
            path = Path(artifact["uri"]).resolve()
            if run_dir not in path.parents or not path.is_file():
                raise click.ClickException(f"invalid artifact path: {path}")
            if path.stat().st_size != artifact["size_bytes"]:
                raise click.ClickException(f"artifact size mismatch: {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise click.ClickException(f"artifact checksum mismatch: {path}")
            formats.add(artifact["format"])
            artifact_by_name[path.name] = (path, artifact)
            if artifact["format"] == "mp4":
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,nb_frames",
                        "-of",
                        "json",
                        str(path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
        required_formats = {"json", "npz", "mp4"}
        if not required_formats.issubset(formats):
            raise click.ClickException(
                f"candidate {manifest['run_id']} lacks artifact formats {required_formats - formats}"
            )
        result_path, _ = artifact_by_name.get("result.json", (None, None))
        if result_path is None:
            raise click.ClickException(f"candidate {manifest['run_id']} lacks result.json")
        result = json.loads(result_path.read_text())
        if "diagnostic" in manifest:
            trajectory_path, _ = artifact_by_name.get("trajectory.npz", (None, None))
            summary_path, _ = artifact_by_name.get("diagnostic_summary.json", (None, None))
            contacts_path, _ = artifact_by_name.get("diagnostic_contacts.json", (None, None))
            if trajectory_path is None or summary_path is None or contacts_path is None:
                raise click.ClickException(
                    f"diagnostic candidate {manifest['run_id']} lacks trajectory/summary/contact artifacts"
                )
            summary = json.loads(summary_path.read_text())
            with np.load(trajectory_path, allow_pickle=False) as archive:
                validate_diagnostic_trajectory(
                    archive,
                    int(result["episode_length"]),
                    summary=summary,
                )
            if int(manifest["diagnostic"].get("episode_steps", -1)) != int(result["episode_length"]):
                raise click.ClickException(
                    f"diagnostic episode length mismatch: {manifest['run_id']}"
                )
            case_list_uri = manifest.get("diagnostic_case", {}).get("case_list_uri")
            case_list_sha = manifest.get("diagnostic_case", {}).get("case_list_sha256")
            if case_list_uri and case_list_sha:
                case_path = Path(case_list_uri).resolve()
                if not case_path.is_file() or sha256_file(case_path) != case_list_sha:
                    raise click.ClickException(f"diagnostic case-list checksum mismatch: {case_path}")
    click.echo(f"validated {len(results)} completed candidate manifests under {run_dir}")


if __name__ == "__main__":
    main()
