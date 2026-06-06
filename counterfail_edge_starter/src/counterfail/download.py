import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="guardian_light")
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--token", type=str, default=None, help="HF token. Usually not needed after `huggingface-cli login`.")
    parser.add_argument("--resume", action="store_true")
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
        print(f"Done: {local_dir}")


if __name__ == "__main__":
    main()
