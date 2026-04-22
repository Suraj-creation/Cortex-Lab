"""Build Cortex modelpack release manifest entries.

Usage:
  python scripts/modelpacks/build_modelpack.py --input-dir <dir> --pack-id gemma-3n
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_entry(pack_id: str, input_dir: Path) -> Dict[str, object]:
    files: List[Dict[str, object]] = []
    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(input_dir).as_posix()
        files.append(
            {
                "path": rel,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )

    return {
        "id": pack_id,
        "display_name": pack_id,
        "version": "0.1.0",
        "target": "generic",
        "requires": [],
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build modelpack manifest entry")
    parser.add_argument("--input-dir", required=True, help="Directory containing modelpack files")
    parser.add_argument("--pack-id", required=True, help="Stable modelpack identifier")
    parser.add_argument("--output", default="infra/modelpacks/release-manifest.json", help="Manifest output path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_path = Path(args.output)
    if output_path.exists():
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": "1.0",
            "generated_at": "",
            "signature_required": True,
            "packs": [],
        }

    pack_entry = build_entry(args.pack_id, input_dir)
    packs = [pack for pack in manifest.get("packs", []) if pack.get("id") != args.pack_id]
    packs.append(pack_entry)

    manifest["packs"] = packs
    manifest["generated_at"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest with {len(packs)} pack(s) to {output_path}")


if __name__ == "__main__":
    main()
