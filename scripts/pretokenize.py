#!/usr/bin/env python3
"""
Pre-tokenize FineWeb-Edu JSONL chunks into GPT-2 binary format.

Reads *.chunk.*.jsonl files, tokenizes with GPT-2, writes .gpt2.bin files.
Original JSONL files are left unchanged.

Binary format: [doc_len: uint32][doc_tokens: uint16 × len] per document.
Supports multi-worker: use --rank and --world-size to process different chunks.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# Add lingua_modified to path for tokenizer imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lingua_modified"))
from lingua.tokenizer import GPT2Tokenizer


def get_text_from_line(line: dict) -> str:
    """Extract text from JSONL line (supports 'text' or 'content' key)."""
    if "text" in line:
        return line["text"]
    if "content" in line:
        return line["content"]
    raise ValueError(f"JSON line must contain 'text' or 'content' key: {list(line.keys())}")


def process_chunk(
    jsonl_path: Path,
    gpt2_tokenizer: GPT2Tokenizer,
    add_bos: bool,
    add_eos: bool,
    force: bool,
) -> int | None:
    """Process one chunk file: tokenize, write .gpt2.bin. Returns token count or None if skipped."""
    gpt2_bin_path = jsonl_path.with_suffix(".gpt2.bin")

    if not force and gpt2_bin_path.exists():
        return None  # Already processed

    total_gpt2_tokens = 0

    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        with open(gpt2_bin_path, "wb") as f_gpt2:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {jsonl_path}: {e}", file=sys.stderr)
                    continue

                text = get_text_from_line(doc)

                if not text:
                    continue

                tokens_gpt2 = gpt2_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

                len_gpt2 = struct.pack("<I", len(tokens_gpt2))
                f_gpt2.write(len_gpt2)
                f_gpt2.write(struct.pack(f"<{len(tokens_gpt2)}H", *tokens_gpt2))

                total_gpt2_tokens += len(tokens_gpt2)

    return total_gpt2_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize FineWeb-Edu JSONL chunks to GPT-2 binary format."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        default=Path("/scratch/zemlians/physics4lm/fineweb_edu/fineweb_edu"),
        nargs="?",
        help="Directory containing *.chunk.*.jsonl files",
    )
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Do not add BOS token",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Do not add EOS token",
    )
    parser.add_argument(
        "--pattern",
        default="*.chunk.*.jsonl",
        help="Glob pattern for JSONL files (default: *.chunk.*.jsonl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .gpt2.bin files",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Worker rank (0 to world_size-1). Chunks assigned: rank, rank+world_size, ...",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of workers. Use with --rank for parallel processing.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    all_chunks = sorted(data_dir.glob(args.pattern))
    if not all_chunks:
        print(f"Error: No files matching {args.pattern} in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Assign chunks to this worker (round-robin)
    chunks = [all_chunks[i] for i in range(args.rank, len(all_chunks), args.world_size)]
    if not chunks:
        print(f"Rank {args.rank}: No chunks assigned (world_size={args.world_size})")
        return

    print(
        f"Rank {args.rank}/{args.world_size}: processing {len(chunks)}/{len(all_chunks)} chunks"
    )
    add_bos = not args.no_bos
    add_eos = not args.no_eos

    gpt2_tokenizer = GPT2Tokenizer()

    total_gpt2_tokens = 0

    for i, chunk_path in enumerate(chunks):
        result = process_chunk(
            chunk_path,
            gpt2_tokenizer,
            add_bos=add_bos,
            add_eos=add_eos,
            force=args.force,
        )
        if result is None:
            print(f"[{i + 1}/{len(chunks)}] Skipping {chunk_path.name} (already processed)")
            continue
        gt = result
        print(f"[{i + 1}/{len(chunks)}] Processed {chunk_path.name}: gpt2={gt:,}")
        total_gpt2_tokens += gt

    print(f"Done. Total tokens: gpt2={total_gpt2_tokens:,}")


if __name__ == "__main__":
    main()
