from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def bind_run_config(template: Path, output: Path, *, code_commit: str) -> str:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ValueError("code_commit must be a full lowercase Git SHA-1")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen run config: {output}")
    payload = json.loads(template.read_text(encoding="utf-8"))
    if payload.get("code_commit") != "BIND_AT_RUN":
        raise ValueError("template code_commit must be BIND_AT_RUN")
    payload["code_commit"] = code_commit
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, output)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind a tracked template to the current clean code commit"
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path("/share/jhk/MobiWAM/Mobipi")
    )
    args = parser.parse_args()
    status = subprocess.check_output(
        ["git", "-C", str(args.repo), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("run config binding requires a clean code tree")
    commit = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    checksum = bind_run_config(args.template, args.output, code_commit=commit)
    print(
        json.dumps(
            {
                "code_commit": commit,
                "config_sha256": checksum,
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
