#!/usr/bin/env bash
set -euo pipefail

DEST="data/raw_datasets/pascalraw_full_archive"
PART_COUNT=2
DRY_RUN=0
JOBS=4
SEGMENT_MIB=128
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
Usage: scripts/download_pascalraw_full_parts.sh [--parts N] [--dest DIR] [--jobs N] [--segment-mib N] [--dry-run]

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
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --segment-mib)
      SEGMENT_MIB="$2"
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
if [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be >= 1" >&2
  exit 2
fi
if [[ "$SEGMENT_MIB" -lt 1 ]]; then
  echo "--segment-mib must be >= 1" >&2
  exit 2
fi

SEGMENT_BYTES=$(( SEGMENT_MIB * 1024 * 1024 ))

mkdir -p "$DEST"
LOCK_DIR="${DEST}/.download.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another PASCALRAW download appears to be active: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

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

download_segment() {
  local url="$1"
  local segment_path="$2"
  local start="$3"
  local end="$4"
  local expected="$5"
  local actual range_start attempt append_path

  actual="$(file_size "$segment_path")"
  if [[ "$actual" == "$expected" ]]; then
    return 0
  fi
  if [[ "$actual" -gt "$expected" ]]; then
    rm -f "$segment_path"
    actual=0
  fi

  for attempt in 1 2 3 4 5; do
    actual="$(file_size "$segment_path")"
    if [[ "$actual" == "$expected" ]]; then
      return 0
    fi
    if [[ "$actual" -gt "$expected" ]]; then
      rm -f "$segment_path"
      actual=0
    fi

    range_start=$(( start + actual ))
    append_path="${segment_path}.append.$$"
    rm -f "$append_path"
    curl \
      --silent \
      --show-error \
      --location \
      --fail \
      --retry 10 \
      --retry-delay 5 \
      --connect-timeout 30 \
      --speed-time 300 \
      --speed-limit 1024 \
      --range "${range_start}-${end}" \
      --output "$append_path" \
      "$url"
    cat "$append_path" >> "$segment_path"
    rm -f "$append_path"
  done

  actual="$(file_size "$segment_path")"
  [[ "$actual" == "$expected" ]]
}

split_existing_partial() {
  local partial="$1"
  local segment_dir="$2"
  local partial_size

  if [[ ! -f "$partial" ]]; then
    return 0
  fi
  if compgen -G "${segment_dir}/segment_*" >/dev/null; then
    return 0
  fi

  partial_size="$(file_size "$partial")"
  if [[ "$partial_size" == "0" ]]; then
    rm -f "$partial"
    return 0
  fi

  log "split existing partial into segments partial=${partial} size=${partial_size}"
  mkdir -p "$segment_dir"
  split -b "$SEGMENT_BYTES" -d -a 6 "$partial" "${segment_dir}/segment_"
  rm -f "$partial"
}

recover_append_files() {
  local segment_dir="$1"
  local append_path base_path

  for append_path in "${segment_dir}"/segment_*.append.*; do
    if [[ ! -f "$append_path" ]]; then
      continue
    fi
    base_path="${append_path%%.append.*}"
    if [[ -f "$base_path" ]]; then
      rm -f "$append_path"
    else
      mv "$append_path" "$base_path"
    fi
  done
}

download_part_parallel() {
  local name="$1"
  local url="$2"
  local expected="$3"
  local final="$4"
  local partial="$5"
  local segment_dir="${final}.segments"
  local segment_count segment_index batch_count pid failures
  local segment_path start end segment_expected actual
  local -a pids

  mkdir -p "$segment_dir"
  recover_append_files "$segment_dir"
  split_existing_partial "$partial" "$segment_dir"

  segment_count=$(( (expected + SEGMENT_BYTES - 1) / SEGMENT_BYTES ))
  log "parallel download ${name} segments=${segment_count} segment_mib=${SEGMENT_MIB} jobs=${JOBS}"

  failures=0
  batch_count=0
  pids=()
  for (( segment_index = 0; segment_index < segment_count; segment_index++ )); do
    segment_path="${segment_dir}/$(printf 'segment_%06d' "$segment_index")"
    start=$(( segment_index * SEGMENT_BYTES ))
    end=$(( start + SEGMENT_BYTES - 1 ))
    if [[ "$end" -ge $(( expected - 1 )) ]]; then
      end=$(( expected - 1 ))
    fi
    segment_expected=$(( end - start + 1 ))
    actual="$(file_size "$segment_path")"
    if [[ "$actual" == "$segment_expected" ]]; then
      continue
    fi

    download_segment "$url" "$segment_path" "$start" "$end" "$segment_expected" &
    pids+=("$!")
    batch_count=$(( batch_count + 1 ))

    if [[ "${#pids[@]}" -ge "$JOBS" ]]; then
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
          failures=$(( failures + 1 ))
        fi
      done
      pids=()
      log "segment batch complete ${name} downloaded_or_checked=${batch_count}/${segment_count} failures=${failures}"
      if [[ "$failures" -ne 0 ]]; then
        return 1
      fi
    fi
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$(( failures + 1 ))
    fi
  done
  if [[ "$failures" -ne 0 ]]; then
    return 1
  fi

  rm -f "$partial"
  for (( segment_index = 0; segment_index < segment_count; segment_index++ )); do
    segment_path="${segment_dir}/$(printf 'segment_%06d' "$segment_index")"
    start=$(( segment_index * SEGMENT_BYTES ))
    end=$(( start + SEGMENT_BYTES - 1 ))
    if [[ "$end" -ge $(( expected - 1 )) ]]; then
      end=$(( expected - 1 ))
    fi
    segment_expected=$(( end - start + 1 ))
    actual="$(file_size "$segment_path")"
    if [[ "$actual" != "$segment_expected" ]]; then
      log "incomplete segment ${name} index=${segment_index} actual=${actual} expected=${segment_expected}"
      return 1
    fi
    cat "$segment_path" >> "$partial"
  done

  actual="$(file_size "$partial")"
  if [[ "$actual" != "$expected" ]]; then
    log "assembled incomplete ${name} actual=${actual} expected=${expected}"
    return 1
  fi
  mv "$partial" "$final"
  rm -rf "$segment_dir"
  log "complete ${name} size=${expected}"
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
    log "dry-run ${name} url=${url} segments=$(( (expected + SEGMENT_BYTES - 1) / SEGMENT_BYTES )) jobs=${JOBS}"
    continue
  fi

  if ! download_part_parallel "$name" "$url" "$expected" "$final" "$partial"; then
    actual="$(file_size "$partial")"
    log "incomplete ${name} actual=${actual} expected=${expected}; leaving partial for resume"
    exit 1
  fi
done

log "requested PASCALRAW full archive parts complete parts=${PART_COUNT} dest=${DEST}"
