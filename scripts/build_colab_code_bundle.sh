#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_NAME="$(basename "$ROOT_DIR")"
TAG="${1:-colab_bundle}"
OUTPUT_DIR="$ROOT_DIR/Local_Output/Colab_Bundles"
STAGING_ROOT="$(mktemp -d /tmp/arc_agi3_colab_bundle.XXXXXX)"
STAGING_PROJECT="$STAGING_ROOT/$PROJECT_NAME"
OUTPUT_ZIP="$OUTPUT_DIR/${TAG}.zip"

mkdir -p "$OUTPUT_DIR"

cleanup() {
  rm -rf "$STAGING_ROOT"
}
trap cleanup EXIT

rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude 'Local_Output' \
  --exclude 'arc-prize-2026-arc-agi-3.zip' \
  "$ROOT_DIR/" "$STAGING_PROJECT/"

rm -f "$OUTPUT_ZIP"
(
  cd "$STAGING_ROOT"
  zip -qry "$OUTPUT_ZIP" "$PROJECT_NAME"
)

echo "Created Colab bundle:"
ls -lh "$OUTPUT_ZIP"
