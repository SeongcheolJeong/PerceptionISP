#!/usr/bin/env python3
"""Download one LIS Google Drive archive with disk and file-size guards."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path


ARCHIVES = {
    "annotations": ("annotations.zip", "1JSlFFUniFe9CG3WPSw8eI4Pk2_U4nTvz"),
    "raw_dark": ("RAW-dark.zip", "1cvR_qmi0jjJ2F5REpnPwJVN13wP_JgJ-"),
    "raw_normal": ("RAW-normal.zip", "1JMwYwKSV8P8WAPZm-D6y_5wpd6vTBy0f"),
    "rgb_dark": ("RGB-dark.zip", "1tbfncHwnXE9Xs3ZWNSS3ZPZB82bsfWA-"),
    "rgb_normal": ("RGB-normal.zip", "1DlY15BXjHUGLCS9sqKE0sdN8BQi7Huaw"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", choices=sorted(ARCHIVES))
    parser.add_argument("--root", default="/Users/seongcheoljeong/PerceptionISP/data/raw_datasets/lis")
    parser.add_argument("--min-free-gib", type=float, default=40.0)
    parser.add_argument("--max-archive-gib", type=float, default=42.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    download_dir = root / "downloads"
    meta_dir = root / "meta"
    download_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    filename, file_id = ARCHIVES[args.archive]
    out = download_dir / filename
    status_path = Path(args.status_json).expanduser() if args.status_json else meta_dir / f"lis_{args.archive}_download_status.json"
    log_path = meta_dir / f"lis_{args.archive}_gdown.log"

    max_bytes = int(args.max_archive_gib * 1024**3)
    min_free_bytes = int(args.min_free_gib * 1024**3)

    if out.exists() and zipfile.is_zipfile(out):
        write_status(status_path, "complete", out, log_path, args, "existing valid zip")
        print(json.dumps({"status": "complete", "path": str(out), "message": "existing valid zip"}, indent=2))
        return 0

    free = shutil.disk_usage(download_dir).free
    if free < min_free_bytes:
        message = f"free disk {free / 1024**3:.2f} GiB is below guard {args.min_free_gib:.2f} GiB"
        write_status(status_path, "blocked_disk", out, log_path, args, message)
        print(json.dumps({"status": "blocked_disk", "message": message}, indent=2))
        return 2

    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--id",
        "--continue",
        file_id,
        "-O",
        str(out),
    ]
    if args.no_resume:
        cmd.remove("--continue")

    with log_path.open("ab") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    status = "running"
    message = "download started"
    while proc.poll() is None:
        size = observed_size(out)
        free = shutil.disk_usage(download_dir).free
        write_status(status_path, status, out, log_path, args, message, pid=proc.pid)
        if size > max_bytes:
            message = f"archive exceeded guard {args.max_archive_gib:.2f} GiB"
            terminate(proc)
            write_status(status_path, "stopped_size_guard", out, log_path, args, message)
            print(json.dumps({"status": "stopped_size_guard", "path": str(out), "message": message}, indent=2))
            return 3
        if free < min_free_bytes:
            message = f"free disk dropped below guard {args.min_free_gib:.2f} GiB"
            terminate(proc)
            write_status(status_path, "stopped_disk_guard", out, log_path, args, message)
            print(json.dumps({"status": "stopped_disk_guard", "path": str(out), "message": message}, indent=2))
            return 4
        time.sleep(max(float(args.poll_seconds), 1.0))

    rc = proc.returncode
    valid_zip = out.exists() and zipfile.is_zipfile(out)
    if rc == 0 and valid_zip:
        status = "complete"
        message = "download completed and zip header is valid"
    elif rc == 0:
        status = "incomplete"
        message = "gdown exited 0 but output is not a valid zip"
    else:
        status = "failed"
        message = f"gdown exited with rc={rc}"
    write_status(status_path, status, out, log_path, args, message)
    print(json.dumps({"status": status, "path": str(out), "message": message}, indent=2))
    return 0 if status == "complete" else 1


def terminate(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def write_status(
    path: Path,
    status: str,
    out: Path,
    log: Path,
    args: argparse.Namespace,
    message: str,
    *,
    pid: int | None = None,
) -> None:
    size = observed_size(out)
    free = shutil.disk_usage(out.parent).free
    payload = {
        "status": status,
        "archive": args.archive,
        "output": str(out),
        "log": str(log),
        "observed_files": observed_files(out),
        "pid": pid,
        "size_bytes": size,
        "size_gib": size / 1024**3,
        "free_gib": free / 1024**3,
        "min_free_gib": args.min_free_gib,
        "max_archive_gib": args.max_archive_gib,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def observed_files(out: Path) -> list[str]:
    paths = []
    if out.exists():
        paths.append(out)
    paths.extend(sorted(out.parent.glob(out.name + "*.part")))
    return [str(path) for path in paths]


def observed_size(out: Path) -> int:
    sizes = []
    if out.exists():
        sizes.append(out.stat().st_size)
    sizes.extend(path.stat().st_size for path in out.parent.glob(out.name + "*.part"))
    return max(sizes) if sizes else 0


if __name__ == "__main__":
    raise SystemExit(main())
