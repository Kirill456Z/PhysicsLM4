"""
Download and prepare the FineWeb-Edu dataset for use with the lingua framework.

Downloads from: HuggingFaceFW/fineweb-edu
Stores to: /scratch/zemlians/physics4lm/fineweb_edu/fineweb_edu/

The output files follow the lingua naming convention:
    fineweb_edu.chunk.XX.jsonl
Each line is a JSON object with a "text" field.

Resume behaviour:
    - Each file is downloaded with hf_hub_download (resume_download=True),
      so partial/interrupted downloads are automatically resumed.
    - The parquet->JSONL conversion skips files that already exist on disk.

Usage:
    python scripts/download_fineweb_edu.py [--fraction 0.5] [--max_workers 16]
"""

import argparse
import glob
import json
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_tree


REPO_ID = "HuggingFaceFW/fineweb-edu"
OUTPUT_DIR = "/scratch/zemlians/physics4lm/fineweb_edu"
SOURCE_NAME = "fineweb_edu"


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def _count_local_parquet_files(local_dir: str) -> tuple:
    """Count already-downloaded parquet files and their total size."""
    files = glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True)
    total_size = sum(os.path.getsize(f) for f in files)
    return len(files), total_size


def _is_file_complete(local_path: str, expected_size: int | None) -> bool:
    """Check if a local file exists and matches expected size (if known)."""
    if not os.path.exists(local_path):
        return False
    if expected_size is None:
        return True
    return os.path.getsize(local_path) == expected_size


def _download_one_file(repo_id: str, rfilename: str, local_dir: str) -> str:
    """Download a single file from HuggingFace Hub. Returns local path."""
    return hf_hub_download(
        repo_id,
        filename=rfilename,
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True,
    )


def download_dataset(
    repo_id: str,
    local_dir: str,
    max_workers: int = 16,
    fraction: float = 1.0,
):
    """Download the FineWeb-Edu dataset from HuggingFace with per-file progress.

    Args:
        repo_id: HuggingFace dataset repo id.
        local_dir: Local directory to store downloaded files.
        max_workers: Number of parallel download threads.
        fraction: Fraction of the dataset to download (0.0-1.0). Default 1.0 (all).
    """
    # ── enumerate remote files ──────────────────────────────────────────
    print(f"Querying remote repository {repo_id} (this may take a few minutes for large repos)...", flush=True)
    remote_files = []
    for entry in list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        if hasattr(entry, "rfilename") and entry.rfilename.endswith(".parquet"):
            remote_files.append(entry)
            if len(remote_files) % 500 == 0:
                print(f"  ... found {len(remote_files)} parquet files so far ...", flush=True)
    remote_files.sort(key=lambda f: f.rfilename)
    total_remote = len(remote_files)
    total_remote_size = sum(
        f.size for f in remote_files if hasattr(f, "size") and f.size
    )
    print(
        f"Remote: {total_remote} parquet files, "
        f"{_format_size(total_remote_size)} total",
        flush=True,
    )

    # ── apply fraction ──────────────────────────────────────────────────
    if fraction < 1.0:
        n_keep = max(1, int(total_remote * fraction))
        remote_files = remote_files[:n_keep]
        subset_size = sum(
            f.size for f in remote_files if hasattr(f, "size") and f.size
        )
        print(
            f"Subset: downloading {n_keep}/{total_remote} files "
            f"({fraction * 100:.0f}%), ~{_format_size(subset_size)}",
            flush=True,
        )

    # ── figure out which files still need downloading ───────────────────
    to_download = []
    already_done = 0
    already_size = 0
    for f in remote_files:
        local_path = os.path.join(local_dir, f.rfilename)
        expected = f.size if hasattr(f, "size") else None
        if _is_file_complete(local_path, expected):
            already_done += 1
            already_size += expected or 0
        else:
            to_download.append(f)

    if already_done:
        print(
            f"Already downloaded: {already_done} files, "
            f"{_format_size(already_size)}",
            flush=True,
        )

    if not to_download:
        print("All requested files are already downloaded.", flush=True)
        return

    need_size = sum(f.size for f in to_download if hasattr(f, "size") and f.size)
    print(
        f"Downloading {len(to_download)} remaining files "
        f"({_format_size(need_size)}) to {local_dir} ...",
        flush=True,
    )

    # ── download with per-file progress ─────────────────────────────────
    download_start = time.time()
    downloaded_bytes = 0
    done_count = 0
    total_to_do = len(to_download)
    max_retries = 5

    for fi, remote_f in enumerate(to_download):
        fname = remote_f.rfilename
        fsize = remote_f.size if hasattr(remote_f, "size") and remote_f.size else 0

        print(
            f"  [{fi + 1}/{total_to_do}] Starting {os.path.basename(fname)} "
            f"({_format_size(fsize)}) ...",
            flush=True,
        )

        for attempt in range(max_retries):
            try:
                _download_one_file(repo_id, fname, local_dir)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 10 * (2 ** attempt)
                    print(
                        f"  ⚠ {os.path.basename(fname)}: attempt {attempt + 1} "
                        f"failed ({e}), retrying in {wait}s ...",
                        flush=True,
                    )
                    time.sleep(wait)
                else:
                    print(f"  ✗ {os.path.basename(fname)}: FAILED after {max_retries} attempts", flush=True)
                    raise

        done_count += 1
        downloaded_bytes += fsize
        elapsed = time.time() - download_start
        speed = downloaded_bytes / elapsed if elapsed > 0 else 0
        remaining_bytes = need_size - downloaded_bytes
        eta = remaining_bytes / speed if speed > 0 else 0

        print(
            f"  [{done_count}/{total_to_do}] Done {os.path.basename(fname)} "
            f"({_format_size(fsize)})  |  "
            f"{_format_size(downloaded_bytes)}/{_format_size(need_size)}  "
            f"({downloaded_bytes * 100 / need_size:.1f}%)  "
            f"speed: {_format_size(int(speed))}/s  ETA: {_format_eta(eta)}",
            flush=True,
        )

    elapsed = time.time() - download_start
    final_count, final_size = _count_local_parquet_files(local_dir)
    print(
        f"\nDownload complete in {_format_eta(elapsed)}: "
        f"{final_count} files, {_format_size(final_size)}",
        flush=True,
    )


def parquet_to_jsonl(parquet_path: str, jsonl_path: str):
    """Convert a single parquet file to JSONL, keeping only the 'text' field."""
    table = pq.read_table(parquet_path, columns=["text"])
    with open(jsonl_path, "w") as f:
        for row in table.to_pydict()["text"]:
            f.write(json.dumps({"text": row}) + "\n")


def prepare_for_lingua(raw_dir: str, output_dir: str):
    """
    Convert downloaded FineWeb-Edu parquet files into lingua-compatible chunked JSONL.

    FineWeb-Edu stores data as .parquet files.
    Lingua expects: output_dir/fineweb_edu/fineweb_edu.chunk.XX.jsonl
    """
    source_dir = os.path.join(output_dir, SOURCE_NAME)
    os.makedirs(source_dir, exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True))

    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found under {raw_dir}/. "
            "Check that the download completed successfully."
        )

    # Determine how many need conversion
    to_convert = []
    skipped = 0
    for idx, pq_file in enumerate(parquet_files):
        out_file = os.path.join(source_dir, f"{SOURCE_NAME}.chunk.{idx:02d}.jsonl")
        if os.path.exists(out_file):
            skipped += 1
        else:
            to_convert.append((idx, pq_file, out_file))

    total = len(parquet_files)
    print(f"Found {total} parquet files: {skipped} already converted, {len(to_convert)} remaining")

    if not to_convert:
        print("All files already converted, nothing to do.")
    else:
        convert_start = time.time()
        for i, (idx, pq_file, out_file) in enumerate(to_convert):
            file_start = time.time()
            pq_size = os.path.getsize(pq_file)
            print(f"  [{i + 1}/{len(to_convert)}] Converting {os.path.basename(pq_file)} "
                  f"({_format_size(pq_size)}) -> {os.path.basename(out_file)}", end="", flush=True)
            parquet_to_jsonl(pq_file, out_file)
            file_elapsed = time.time() - file_start

            # ETA based on average time per file so far
            total_elapsed = time.time() - convert_start
            avg_per_file = total_elapsed / (i + 1)
            remaining = len(to_convert) - (i + 1)
            eta = avg_per_file * remaining
            out_size = os.path.getsize(out_file)
            print(f"  done in {_format_eta(file_elapsed)} "
                  f"(out: {_format_size(out_size)}, ETA: {_format_eta(eta)})")

        total_elapsed = time.time() - convert_start
        print(f"\nConversion finished in {_format_eta(total_elapsed)}")

    final_chunks = sorted(glob.glob(os.path.join(source_dir, f"{SOURCE_NAME}.chunk.*.jsonl")))
    total_jsonl_size = sum(os.path.getsize(f) for f in final_chunks)
    print(f"\nDone! {len(final_chunks)} chunk files ready in {source_dir} ({_format_size(total_jsonl_size)} total)")
    print(f"Lingua root_dir should be set to: {output_dir}")
    print(f"Lingua sources should include: {SOURCE_NAME}")


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu for lingua training")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Number of parallel download workers (default: 16)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of dataset to download, 0.0-1.0 (default: 1.0 = all)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, only run conversion (if files already downloaded)")
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be between 0.0 (exclusive) and 1.0 (inclusive)")

    raw_download_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_download_dir, exist_ok=True)

    if not args.skip_download:
        download_dataset(
            REPO_ID,
            raw_download_dir,
            max_workers=args.max_workers,
            fraction=args.fraction,
        )

    prepare_for_lingua(raw_download_dir, args.output_dir)


if __name__ == "__main__":
    main()
