"""Download configured VLM repository snapshots without loading model weights."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import snapshot_download


CONFIG_PATH = Path(__file__).resolve().parents[1] / "src" / "tasks" / "config" / "config.json"


def load_profiles() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config.get("vlm", {}).get("models", {})


def resolve_profile_id(requested: str, profiles: dict) -> Optional[str]:
    if requested in profiles:
        return requested
    return next(
        (profile_id for profile_id, profile in profiles.items()
         if profile.get("repo_id") == requested and requested),
        None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download VLM weights to the Hugging Face cache without loading them into RAM or GPU."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Profile ID or repository ID; repeat to download selected profiles. Default: all configured profiles.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory. Defaults to the standard local cache.",
    )
    parser.add_argument("--revision", help="Optional Hugging Face branch, tag, or commit to download.")
    parser.add_argument("--dry-run", action="store_true", help="List downloads and skipped profiles without network access.")
    args = parser.parse_args(argv)

    profiles = load_profiles()
    if not profiles:
        parser.error(f"No VLM profiles found in {CONFIG_PATH}")

    requested_profiles = args.models or list(profiles)
    failures = []
    skipped = []

    for requested in requested_profiles:
        profile_id = resolve_profile_id(requested, profiles)
        if profile_id is None:
            print(f"ERROR: Unknown profile or repository ID: {requested}")
            failures.append(requested)
            continue

        profile = profiles[profile_id]
        repo_id = (profile.get("repo_id") or "").strip()
        if not repo_id:
            message = f"SKIP: {profile_id} has no repo_id in {CONFIG_PATH}"
            print(message)
            skipped.append(profile_id)
            if args.models:
                failures.append(profile_id)
            continue

        label = profile.get("label", profile_id)
        if args.dry_run:
            note = " (not video-capable)" if profile.get("supports_video") is False else ""
            print(f"READY: {label} [{profile_id}] -> {repo_id}{note}")
            continue

        note = " (not video-capable)" if profile.get("supports_video") is False else ""
        print(f"Downloading {label} [{profile_id}] from {repo_id}{note}...")
        try:
            download_patterns = profile.get("download_patterns")
            if download_patterns:
                print(f"Including files: {', '.join(download_patterns)}")
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                revision=args.revision,
                allow_patterns=download_patterns,
            )
            print(f"Cached at: {snapshot_path}")
        except Exception as exc:
            print(f"ERROR: Could not download {repo_id}: {exc}")
            failures.append(profile_id)

    if skipped:
        print(f"Skipped profiles without repository IDs: {', '.join(skipped)}")
    if failures:
        print(f"Download completed with {len(failures)} error(s).")
        return 1
    if args.dry_run:
        print("Dry run complete; no files downloaded.")
    else:
        print("All configured repositories with IDs are cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())