#!/usr/bin/env python3
"""Inspect the keys, shapes, and spaces of one or more D4RL datasets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Optional, Sequence

import d4rl  # noqa: F401 - importing D4RL registers its Gym environments
import gym
import numpy as np


DEFAULT_ENVIRONMENTS = ("maze2d-umaze-v1", "hopper-medium-v2")


def describe_dataset(env_name: str) -> None:
    if not env_name.strip():
        raise ValueError("environment name must not be empty")

    try:
        env = gym.make(env_name)
    except Exception as error:
        raise RuntimeError(f"could not create D4RL environment {env_name!r}") from error

    try:
        dataset = d4rl.qlearning_dataset(env)
        if not isinstance(dataset, Mapping):
            raise TypeError(
                f"D4RL returned {type(dataset).__name__}, expected a mapping"
            )

        print("-" * 72)
        print(f"Environment: {env_name}")
        print(f"Dataset type: {type(dataset).__name__}")
        print(f"Dataset keys: {list(dataset)}")
        for key, value in dataset.items():
            array = np.asarray(value)
            print(f"  {key}: shape={array.shape} dtype={array.dtype}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "environments",
        nargs="*",
        default=list(DEFAULT_ENVIRONMENTS),
        help="D4RL Gym environment IDs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    for env_name in args.environments:
        describe_dataset(env_name)


if __name__ == "__main__":
    main()
