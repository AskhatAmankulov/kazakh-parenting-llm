"""Config loading with dotted CLI overrides: --set train.epochs=2 model.base=Qwen/Qwen2.5-7B-Instruct"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Cfg(dict):
    """dict with attribute access (cfg.train.lr) that stays a plain dict for yaml/json dumping."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) else value


def _parse_scalar(raw: str) -> Any:
    # yaml gives us ints, floats, bools, null and [lists] for free
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Cfg(apply_overrides(cfg, overrides or []))


def resolve(p: str | Path) -> Path:
    """Paths in the config are relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
