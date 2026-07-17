from __future__ import annotations

import argparse

from ..config import load_config


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def parse_config(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    return args, load_config(args.config, args.overrides)
