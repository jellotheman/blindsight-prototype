"""Replay one retained capture or a selected set through a chosen production provider."""

from __future__ import annotations

import argparse
from pathlib import Path

from blindsight.media_urls import ModalMediaUrlStore
from blindsight.providers import GeminiAdapter, RekaChatAdapter, SingleProvider
from blindsight.replay import ReplayService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-judge retained captures without mutating accepted runs or live sessions."
    )
    parser.add_argument("--evidence-root", type=Path, default=Path("runs"))
    parser.add_argument("--capture-id", action="append", dest="capture_ids")
    parser.add_argument("--all", action="store_true", help="Replay every retained capture.")
    parser.add_argument("--provider", choices=("reka", "gemini"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--public-base-url", help="Deployed BlindSight HTTPS URL for Reka media.")
    parser.add_argument("--modal-dict", default="blindsight-capture-state")
    args = parser.parse_args()

    if args.all == bool(args.capture_ids):
        parser.error("choose either --all or one or more --capture-id values")

    if args.provider == "gemini":
        provider = SingleProvider(GeminiAdapter(model=args.model))
    else:
        if not args.public_base_url or not args.public_base_url.startswith("https://"):
            parser.error("Reka replay requires --public-base-url with the deployed HTTPS URL")
        import modal

        dictionary = modal.Dict.from_name(args.modal_dict, create_if_missing=False)
        media_urls = ModalMediaUrlStore(dictionary, args.public_base_url)
        provider = SingleProvider(RekaChatAdapter(model=args.model), media_urls=media_urls)

    capture_ids = args.capture_ids
    if args.all:
        capture_ids = sorted(
            path.name for path in args.evidence_root.iterdir() if (path / "capture.json").exists()
        )
    assert capture_ids is not None
    for output in ReplayService(provider).replay_many(args.evidence_root, capture_ids):
        print(output)


if __name__ == "__main__":
    main()
