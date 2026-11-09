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
    """Download `subfolder/*` of `repo` and return the adapter folder path."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    dest = adapter_dir(data_dir, repo, subfolder)
    if dest.is_dir() and any(dest.iterdir()):
        return str(dest)
    allow = [f"{subfolder}/*"] if subfolder else None
    try:
        snapshot_download(
            repo_id=repo,
            allow_patterns=allow,
            token=token or None,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
    except HfHubHTTPError as e:
        log.error("adapter download refused for %s: %s\n%s", repo, e, traceback.format_exc())
        code = e.response.status_code if e.response is not None else 0
        if code == 401:
            raise FetchError(HF_TOKEN_MESSAGE) from e
        raise FetchError(DOWNLOAD_FAILED) from e
    except Exception as e:  # noqa: BLE001 - plain error to the client, traceback to the log
        log.error("adapter download failed for %s: %s\n%s", repo, e, traceback.format_exc())
        raise FetchError(DOWNLOAD_FAILED) from e
    return str(dest)
