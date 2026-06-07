import argparse
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

from .paths import resolve_output_path

PRESETS = {
    "guardian_light": [
        "paulpacaud/bdv2fail_train_dataset",
        "paulpacaud/bdv2fail_val_dataset",
        "paulpacaud/bdv2fail_test_dataset",
        "paulpacaud/rlbenchfail_test_dataset",
        "paulpacaud/ur5fail_test_dataset",
    ],
    "guardian_full": [
        "paulpacaud/bdv2fail_train_dataset",
        "paulpacaud/bdv2fail_val_dataset",
        "paulpacaud/bdv2fail_test_dataset",
        "paulpacaud/rlbenchfail_train_dataset",
        "paulpacaud/rlbenchfail_val_dataset",
        "paulpacaud/rlbenchfail_test_dataset",
        "paulpacaud/ur5fail_train_dataset",
        "paulpacaud/ur5fail_val_dataset",
        "paulpacaud/ur5fail_test_dataset",
    ],
    # Optional and large. The Hugging Face page reports ~90GB.
    "bridge_original": [
        "nvidia/bridge_lerobot_v3",
    ],
}


def extract_tar_archives(local_dir: Path) -> None:
    """Extract all .tar.gz archives found in the downloaded dataset directory."""
    for tar_path in sorted(local_dir.glob("*.tar.gz")):
        print(f"  Extracting {tar_path.name} ({tar_path.stat().st_size / 1e6:.1f} MB)...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=local_dir)
        print(f"  Extracted {tar_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="guardian_light")
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--token", type=str, default=None, help="HF token. Usually not needed after `huggingface-cli login`.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_extract", action="store_true", help="Skip automatic extraction of tar archives.")
    args = parser.parse_args()

    out_dir = resolve_output_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in PRESETS[args.preset]:
        local_name = repo_id.replace("/", "__")
        local_dir = out_dir / local_name
        print(f"\nDownloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=args.token,
            resume_download=args.resume,
            local_dir_use_symlinks=False,
        )
        print(f"Done downloading: {local_dir}")

        if not args.no_extract:
            extract_tar_archives(local_dir)

    print("\nAll datasets ready.")


if __name__ == "__main__":
    main()

