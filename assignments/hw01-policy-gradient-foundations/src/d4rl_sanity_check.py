#!/usr/bin/env python3
"""Run the D4RL dataset sanity check without dumping an entire dataset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any, Optional, Sequence

import d4rl  # noqa: F401 - importing D4RL registers its Gym environments
import gym
import numpy as np

MAX_SEED = 2**32 - 1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"value must be between 0 and {MAX_SEED}")
    return parsed


def reset_env(env: Any, seed: Optional[int]) -> Any:
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def summarize_dataset(label: str, dataset: Any, sample_rows: int) -> None:
    if not isinstance(dataset, Mapping):
        raise TypeError(f"{label} dataset is not a mapping")

    print(f"{label} keys: {list(dataset)}")
    for key, value in dataset.items():
        array = np.asarray(value)
        print(f"  {key}: shape={array.shape} dtype={array.dtype}")
        if key == "observations":
            print(f"  first observations:\n{array[:sample_rows]}")


def run_sanity_check(env_name: str, seed: int, sample_rows: int) -> None:
    if not env_name.strip():
        raise ValueError("environment name must not be empty")

    try:
        env = gym.make(env_name)
    except Exception as error:
        raise RuntimeError(f"could not create D4RL environment {env_name!r}") from error

    try:
        reset_env(env, seed)
        env.step(env.action_space.sample())

        get_dataset = getattr(env, "get_dataset", None)
        if get_dataset is None:
            raise AttributeError(f"{env_name!r} does not expose get_dataset()")

        raw_dataset = get_dataset()
        qlearning_dataset = d4rl.qlearning_dataset(env)
        print(f"Environment: {env_name}")
        summarize_dataset("Raw", raw_dataset, sample_rows)
        summarize_dataset("Q-learning", qlearning_dataset, sample_rows)
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", nargs="?", default="hopper-random-v2")
    parser.add_argument("--seed", type=seed_value, default=10)
    parser.add_argument("--sample-rows", type=positive_int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_sanity_check(args.environment, args.seed, args.sample_rows)


if __name__ == "__main__":
    main()
