#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="data/raw_datasets/pascalraw_full_archive"
DEST="data/raw_datasets/pascalraw_full_extract"
COUNT=512
START_ID=1
PART_COUNT=6
PART_SUFFIXES=(aa ab ac ad ae af ag ah ai aj ak al am an)

usage() {
  cat <<'EOF'
Usage: scripts/extract_pascalraw_native_subset.sh [--count N] [--start-id N] [--parts N] [--archive-dir DIR] [--dest DIR]

Extracts contiguous PASCALRAW original NEF + JPG files from the downloaded split
PASCALRAW.tar.gz* archive prefix. A truncated gzip/tar warning is expected when
only an archive prefix has been downloaded; the script succeeds if all requested
NEF and JPG files are present after extraction.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-dir)
      ARCHIVE_DIR="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --start-id)
      START_ID="$2"
      shift 2
      ;;
    --parts)
      PART_COUNT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$COUNT" -lt 1 ]]; then
  echo "--count must be >= 1" >&2
  exit 2
fi
if [[ "$START_ID" -lt 1 ]]; then
  echo "--start-id must be >= 1" >&2
  exit 2
fi
if [[ "$PART_COUNT" -lt 1 || "$PART_COUNT" -gt "${#PART_SUFFIXES[@]}" ]]; then
  echo "--parts must be between 1 and ${#PART_SUFFIXES[@]}" >&2
  exit 2
fi

mkdir -p "$DEST"

part_paths=()
for (( index = 0; index < PART_COUNT; index++ )); do
  suffix="${PART_SUFFIXES[$index]}"
  path="${ARCHIVE_DIR}/PASCALRAW.tar.gz${suffix}"
  if [[ ! -f "$path" ]]; then
    echo "Missing archive part: $path" >&2
    exit 1
  fi
  part_paths+=("$path")
done

members=()
end_id=$(( START_ID + COUNT - 1 ))
for (( id = START_ID; id <= end_id; id++ )); do
  sample_id="$(printf '2014_%06d' "$id")"
  members+=("PASCALRAW/original/jpg/${sample_id}.jpg")
  members+=("PASCALRAW/original/raw/${sample_id}.nef")
done

gzip_err="$(mktemp /tmp/pascalraw_extract_gzip.XXXXXX)"
tar_err="$(mktemp /tmp/pascalraw_extract_tar.XXXXXX)"
set +e
cat "${part_paths[@]}" \
  | gzip -cd 2>"$gzip_err" \
  | tar -xf - -C "$DEST" "${members[@]}" 2>"$tar_err"
extract_rc=$?
set -e

raw_dir="${DEST}/PASCALRAW/original/raw"
jpg_dir="${DEST}/PASCALRAW/original/jpg"
missing=0
for (( id = START_ID; id <= end_id; id++ )); do
  sample_id="$(printf '2014_%06d' "$id")"
  [[ -f "${raw_dir}/${sample_id}.nef" ]] || missing=$(( missing + 1 ))
  [[ -f "${jpg_dir}/${sample_id}.jpg" ]] || missing=$(( missing + 1 ))
done

if [[ "$missing" -ne 0 ]]; then
  echo "Extraction incomplete: missing_files=${missing} extract_rc=${extract_rc}" >&2
  echo "gzip stderr: $gzip_err" >&2
  echo "tar stderr: $tar_err" >&2
  exit 1
fi

echo "Extracted PASCALRAW native subset count=${COUNT} start_id=${START_ID} end_id=${end_id} parts=${PART_COUNT} dest=${DEST}"
if [[ "$extract_rc" -ne 0 ]]; then
  echo "Note: archive prefix ended with expected gzip/tar warning; requested files are complete."
  echo "gzip stderr: $gzip_err"
  echo "tar stderr: $tar_err"
fi
