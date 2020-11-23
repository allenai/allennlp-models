#!/usr/bin/env python3

import argparse
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("version_type", choices=["stable", "latest", "current"])
    parser.add_argument("--minimal", action="store_true", default=False)
    parser.add_argument("--as-range", action="store_true", default=False)
    return parser.parse_args()


def post_process(version: str, minimal: bool = False, as_range: bool = False):
    assert not (minimal and as_range)
    if version.startswith("v"):
        version = version[1:]
    if as_range:
        major, minor, *_ = version.split(".")
        return f">={version},<{major}.{int(minor)+1}"
    return version if minimal else f"v{version}"


def get_current_version() -> str:
    VERSION: Dict[str, str] = {}
    with open("allennlp_models/version.py", "r") as version_file:
        exec(version_file.read(), VERSION)
    return VERSION["VERSION"]


def get_latest_version() -> str:
    # Import this here so this requirements isn't mandatory when we just want to
    # call `get_current_version`.
    import requests

    resp = requests.get("https://api.github.com/repos/allenai/allennlp-models/tags")
    return resp.json()[0]["name"]


def get_stable_version() -> str:
    import requests

    resp = requests.get("https://api.github.com/repos/allenai/allennlp-models/releases/latest")
    return resp.json()["tag_name"]


def main() -> None:
    opts = parse_args()
    if opts.version_type == "stable":
        print(post_process(get_stable_version(), opts.minimal, opts.as_range))
    elif opts.version_type == "latest":
        print(post_process(get_latest_version(), opts.minimal, opts.as_range))
    elif opts.version_type == "current":
        print(post_process(get_current_version(), opts.minimal, opts.as_range))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
