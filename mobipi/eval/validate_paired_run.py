"""Validate a completed paired-rollout directory without loading the policy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click

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
        for artifact in manifest["artifacts"]:
            path = Path(artifact["uri"]).resolve()
            if run_dir not in path.parents or not path.is_file():
                raise click.ClickException(f"invalid artifact path: {path}")
            if path.stat().st_size != artifact["size_bytes"]:
                raise click.ClickException(f"artifact size mismatch: {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise click.ClickException(f"artifact checksum mismatch: {path}")
            formats.add(artifact["format"])
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
    click.echo(f"validated {len(results)} completed candidate manifests under {run_dir}")


if __name__ == "__main__":
    main()
