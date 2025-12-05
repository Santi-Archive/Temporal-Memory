from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Tuple

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    print("huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
SENTENCE_TRANSFORMER_ID = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_MODEL_ID = "dbmdz/bert-large-cased-finetuned-conll03-english"
SPACY_MODEL = "en_core_web_sm"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_sentence_transformer() -> Tuple[bool, str]:
    target_root = MODELS_DIR / "sentence_transformers"
    ensure_dir(target_root)
    print("=" * 60)
    print(f"Downloading Sentence Transformer model: {SENTENCE_TRANSFORMER_ID}")
    print(f"Target directory: {target_root}")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=SENTENCE_TRANSFORMER_ID,
            local_dir=target_root,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True, f"✓ Sentence Transformer stored in {target_root}"
    except Exception as exc:  # noqa: BLE001
        return False, f"✗ Failed to download Sentence Transformer: {exc}"


def download_huggingface_model() -> Tuple[bool, str]:
    target_root = MODELS_DIR / "huggingface"
    ensure_dir(target_root)
    print("=" * 60)
    print(f"Downloading HuggingFace SRL model: {HUGGINGFACE_MODEL_ID}")
    print(f"Target directory: {target_root}")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=HUGGINGFACE_MODEL_ID,
            local_dir=target_root,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True, f"✓ HuggingFace model stored in {target_root}"
    except Exception as exc:  # noqa: BLE001
        return False, f"✗ Failed to download HuggingFace model: {exc}"


def download_spacy_model() -> Tuple[bool, str]:
    """Install spaCy model (stored in spaCy's standard location)."""
    print("=" * 60)
    print(f"Installing spaCy model: {SPACY_MODEL}")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
            check=True,
            capture_output=False,
        )
        return True, "✓ spaCy model installed (managed by spaCy)"
    except subprocess.CalledProcessError as exc:
        return False, f"✗ Failed to install spaCy model: {exc}"


def main() -> None:
    print("\n" + "=" * 60)
    print("Temporal Memory Layer - Model Download")
    print("=" * 60)
    print("Models will be stored under: models/\n")

    ensure_dir(MODELS_DIR)

    successes = []

    success, message = download_sentence_transformer()
    print(message)
    successes.append(success)

    success, message = download_huggingface_model()
    print(message)
    successes.append(success)

    success, message = download_spacy_model()
    print(message)
    successes.append(success)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if all(successes):
        print("✓ All models downloaded successfully.")
    else:
        print("⚠ Some models failed to download. See messages above.")

    print("\nModels directory contents:")
    for sub in MODELS_DIR.iterdir():
        if sub.is_dir():
            print(f" - {sub.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted.")
        sys.exit(1)

