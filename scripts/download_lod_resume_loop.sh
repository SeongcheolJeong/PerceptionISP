#!/usr/bin/env bash
set -u

ROOT="${PERCEPTION_ISP_ROOT:-/Users/seongcheoljeong/PerceptionISP}"
DOWNLOAD_DIR="$ROOT/data/raw_datasets/lod/downloads"
URL="${LOD_GDRIVE_URL:-https://drive.google.com/file/d/1Jkm4mvynWxc7lXSH3H9sLI0wJ6p6ftvZ/view?usp=sharing}"
OUT_PART="$DOWNLOAD_DIR/LOD_BMVC2021.zip.part"
OUT_ZIP="$DOWNLOAD_DIR/LOD_BMVC2021.zip"
RETRY_SECONDS="${LOD_RETRY_SECONDS:-1800}"
MIN_FREE_KB="${LOD_MIN_FREE_KB:-31457280}"

mkdir -p "$DOWNLOAD_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_zip_valid() {
  local path="$1"
  python3 - "$path" <<'PY'
import sys
import zipfile
path = sys.argv[1]
try:
    ok = zipfile.is_zipfile(path)
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

while true; do
  if [ -f "$OUT_ZIP" ] && is_zip_valid "$OUT_ZIP"; then
    log "complete: $OUT_ZIP is a readable zip archive"
    exit 0
  fi

  if [ -f "$OUT_PART" ] && is_zip_valid "$OUT_PART"; then
    log "partial path is already a readable zip; renaming to final archive"
    mv "$OUT_PART" "$OUT_ZIP"
    exit 0
  fi

  free_kb="$(df -Pk "$DOWNLOAD_DIR" | awk 'NR==2 {print $4}')"
  if [ "${free_kb:-0}" -lt "$MIN_FREE_KB" ]; then
    log "blocked: free disk ${free_kb:-0} KB is below guard $MIN_FREE_KB KB"
    sleep "$RETRY_SECONDS"
    continue
  fi

  if [ -f "$OUT_PART" ]; then
    size="$(du -h "$OUT_PART" | awk '{print $1}')"
    log "resume attempt: $OUT_PART size=$size"
  else
    log "download attempt: $OUT_PART"
  fi

  python3 -m gdown --fuzzy --continue "$URL" -O "$OUT_PART"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    if [ -f "$OUT_PART" ] && is_zip_valid "$OUT_PART"; then
      log "download completed and zip validation passed; renaming to $OUT_ZIP"
      mv "$OUT_PART" "$OUT_ZIP"
      exit 0
    fi
    log "gdown exited successfully, but zip validation has not passed yet; retrying after sleep"
  else
    log "gdown failed with rc=$rc; likely quota/auth/transient network issue"
  fi

  log "sleeping ${RETRY_SECONDS}s before next attempt"
  sleep "$RETRY_SECONDS"
done
