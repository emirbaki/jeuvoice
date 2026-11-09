"""Upload a local checkpoint folder to a private Hugging Face repo.

YOU run this, the agent never uploads. The tr2 adapter lives here after training:
    python engine/tools/upload_adapter.py --dir tts_lab/outputs/omnivoice_tr2/checkpoint-6000 --repo emirbakid/jeuvoice-omnivoice-tr2

Needs a token with write permission: --token or the HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Upload a checkpoint folder to a private Hugging Face repo.")
    ap.add_argument("--dir", required=True, help="local checkpoint folder")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. user/adapter")
    ap.add_argument("--token", default=None, help="write token (else HF_TOKEN)")
    args = ap.parse_args(argv)

    import os

    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Set --token or the HF_TOKEN environment variable.", file=sys.stderr)
        return 2
    folder = Path(args.dir)
    if not folder.is_dir():
        print(f"No folder: {folder}", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    api.create_repo(args.repo, private=True, exist_ok=True)
    print(f"Uploading {folder} to {args.repo} (private)…")
    api.upload_folder(repo_id=args.repo, folder_path=str(folder), repo_type="model")
    print("Done. Check the repo page, then run the Voice tab with the token in Settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
