"""Sign Cortex modelpack release manifest.

This scaffold computes a SHA-256 digest and writes it to a sidecar .sig file.
Replace with asymmetric signing (KMS/HSM) for production distribution.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign modelpack manifest (digest sidecar scaffold)")
    parser.add_argument("--manifest", default="infra/modelpacks/release-manifest.json", help="Manifest path")
    parser.add_argument("--output", default="", help="Optional signature output path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists() or not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output = Path(args.output) if args.output else manifest_path.with_suffix(manifest_path.suffix + ".sig")
    output.write_text(digest + "\n", encoding="utf-8")
    print(f"Wrote manifest signature digest to {output}")


if __name__ == "__main__":
    main()
