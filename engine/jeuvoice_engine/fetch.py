"""Fetch a fine-tuned adapter from Hugging Face into the engine data folder.

Errors are plain words for the UI. The traceback goes only to the log.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

HF_TOKEN_MESSAGE = "Add your Hugging Face token in Settings."
DOWNLOAD_FAILED = "The model files could not be downloaded. Check the connection and try again."


class FetchError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def adapter_dir(data_dir: Path, repo: str, subfolder: str | None) -> Path:
    safe = repo.replace("/", "__")
    return data_dir / "models" / safe / (subfolder or "")


def fetch_adapter(repo: str, subfolder: str | None, token: str | None, data_dir: Path, log: logging.Logger) -> str:
    """Download the adapter and return the folder that holds adapter_config.json.

    `upload_adapter.py` puts the checkpoint files at the repo root, while a
    model repo may keep them in a subfolder. Try the subfolder first, then the
    root, so both layouts work. The old code put `local_dir` on the subfolder
    itself, which made Hugging Face write `subfolder/subfolder/...`.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    root = adapter_dir(data_dir, repo, None)
    candidates = [subfolder] if subfolder else []
    candidates.append(None)
    for sub in candidates:
        allow = [f"{sub}/*"] if sub else None
        try:
            snapshot_download(
                repo_id=repo,
                allow_patterns=allow,
                token=token or None,
                local_dir=str(root),
                local_dir_use_symlinks=False,
            )
        except HfHubHTTPError as e:
            log.error("adapter download refused for %s (%s): %s\n%s", repo, sub, e, traceback.format_exc())
            code = e.response.status_code if e.response is not None else 0
            if code == 401:
                raise FetchError(HF_TOKEN_MESSAGE) from e
            # A missing subfolder is normal; the root may hold the files.
            continue
        except Exception as e:  # noqa: BLE001 - plain error to the client, traceback to the log
            log.error("adapter download failed for %s (%s): %s\n%s", repo, sub, e, traceback.format_exc())
            continue
        dest = root / (sub or "")
        if (dest / "adapter_config.json").is_file():
            return str(dest)
    raise FetchError(DOWNLOAD_FAILED)
