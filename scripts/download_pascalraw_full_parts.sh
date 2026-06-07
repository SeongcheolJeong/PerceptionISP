#!/usr/bin/env bash
set -euo pipefail

DEST="data/raw_datasets/pascalraw_full_archive"
PART_COUNT=2
DRY_RUN=0
BASE_URL="https://stacks.stanford.edu/file/hq050zr7488"
PART_SUFFIXES=(aa ab ac ad ae af ag ah ai aj ak al am an)
PART_SIZES=(
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  10737418240
  8507181589
)

usage() {
  cat <<'EOF'
Usage: scripts/download_pascalraw_full_parts.sh [--parts N] [--dest DIR] [--dry-run]

Downloads the first N split chunks of Stanford PASCALRAW full-resolution RAW archive.
The complete archive is PASCALRAW.tar.gzaa through PASCALRAW.tar.gzan.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parts)
      PART_COUNT="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ "$PART_COUNT" -lt 1 || "$PART_COUNT" -gt "${#PART_SUFFIXES[@]}" ]]; then
  echo "--parts must be between 1 and ${#PART_SUFFIXES[@]}" >&2
  exit 2
fi

mkdir -p "$DEST"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

file_size() {
  if [[ -f "$1" ]]; then
    stat -f '%z' "$1"
  else
    printf '0'
  fi
}

for (( index = 0; index < PART_COUNT; index++ )); do
  suffix="${PART_SUFFIXES[$index]}"
  expected="${PART_SIZES[$index]}"
  name="PASCALRAW.tar.gz${suffix}"
  url="${BASE_URL}/${name}"
  final="${DEST}/${name}"
  partial="${final}.partial"

  if [[ "$(file_size "$final")" == "$expected" ]]; then
    log "skip complete ${name} size=${expected}"
    continue
  fi
  if [[ -f "$final" && "$(file_size "$final")" != "$expected" ]]; then
    log "move incomplete final to partial ${name}"
    mv "$final" "$partial"
  fi

  log "download ${name} expected=${expected} dest=${final}"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  curl \
    --location \
    --fail \
    --continue-at - \
    --retry 20 \
    --retry-delay 10 \
    --connect-timeout 30 \
    --speed-time 300 \
    --speed-limit 1024 \
    --output "$partial" \
    "$url"

  actual="$(file_size "$partial")"
  if [[ "$actual" != "$expected" ]]; then
    log "incomplete ${name} actual=${actual} expected=${expected}; leaving partial for resume"
    exit 1
  fi
  mv "$partial" "$final"
  log "complete ${name} size=${expected}"
done

log "requested PASCALRAW full archive parts complete parts=${PART_COUNT} dest=${DEST}"
