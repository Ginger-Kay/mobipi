#!/usr/bin/env python3
"""Check that RoboCasa XML assets do not reference missing files.

The normal Mobi-pi preflight only checks whether asset directories are
non-empty. This audit follows every XML ``file=`` reference so a rollout is
never started with a partially downloaded fixture archive.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/share/jhk/MobiWAM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_reference(xml_path: Path, raw: str, asset_root: Path) -> Path | None:
    value = raw.strip()
    if not value or "://" in value:
        return None
    # RoboCasa XML sometimes stores an asset-root-relative path.
    normalized = value.replace("\\", "/")
    if normalized.startswith("models/assets/"):
        return asset_root / normalized[len("models/assets/") :]
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return xml_path.parent / candidate


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if root != DEFAULT_ROOT.resolve():
        raise RuntimeError(f"Project root must be exactly {DEFAULT_ROOT}: {root}")
    repo = root / "Mobipi"
    asset_root = repo / "external" / "robocasa" / "robocasa" / "models" / "assets"
    output = args.output or root / "artifacts" / "MMWAM-OBC-001" / "setup" / "robocasa_asset_integrity.json"

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "asset_root": str(asset_root),
        "xml_files": 0,
        "references": 0,
        "missing": [],
        "parse_errors": [],
    }
    if not asset_root.is_dir():
        report["parse_errors"].append(f"asset root is missing: {asset_root}")
    else:
        for xml_path in sorted(asset_root.rglob("*.xml")):
            report["xml_files"] += 1
            try:
                tree = ET.parse(xml_path)
            except (ET.ParseError, OSError) as exc:
                report["parse_errors"].append(
                    {"file": str(xml_path), "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            for element in tree.iter():
                raw = element.attrib.get("file")
                if raw is None:
                    continue
                report["references"] += 1
                resolved = resolve_reference(xml_path, raw, asset_root)
                if resolved is not None and not resolved.is_file():
                    report["missing"].append(
                        {
                            "xml": str(xml_path),
                            "attribute": "file",
                            "reference": raw,
                            "resolved": str(resolved),
                        }
                    )

    report["missing_count"] = len(report["missing"])
    report["parse_error_count"] = len(report["parse_errors"])
    report["status"] = "pass" if not report["missing"] and not report["parse_errors"] else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"asset_integrity {report['status']}: "
        f"xml={report['xml_files']} refs={report['references']} "
        f"missing={report['missing_count']} output={output}"
    )
    for item in report["missing"][:20]:
        print(f"MISSING: {item['resolved']}")
    for item in report["parse_errors"][:20]:
        print(f"PARSE_ERROR: {item}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
