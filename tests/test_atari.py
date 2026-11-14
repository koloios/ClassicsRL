import hashlib
import os
from dataclasses import replace

import numpy as np
import pytest

from classicsrl.config import Config
from classicsrl.envs import make_env
from classicsrl.replay import ReplayBuffer

pytestmark = [
    pytest.mark.atari,
    pytest.mark.skipif(os.getenv("CLASSICSRL_TEST_ATARI") != "1", reason="opt-in Atari smoke test"),
]


def test_pong_protocol_counts_raw_frames_and_preserves_timeout_observation():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")
    config = replace(Config(), noop_max=0, fire_on_reset=False, max_episode_steps=2)
    env = make_env(config, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=42)
        assert observation.shape == (4, 84, 84) and observation.dtype == np.uint8
        assert np.array_equal(observation[0], observation[-1])
        assert env.get_wrapper_attr("raw_frames") == 0
        env.step(0)
        final, _, terminated, truncated, _ = env.step(0)
        assert not terminated and truncated
        assert env.get_wrapper_attr("raw_frames") == 8
        assert final.shape == (4, 84, 84)
        assert env.render().shape == (210, 160, 3)
    finally:
        env.close()


def test_real_pong_replay_stacks_match_environment_across_resets_and_wraparound():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")
    env = make_env(replace(Config(), noop_max=0, max_episode_steps=7))
    replay = ReplayBuffer(17)
    rng = np.random.default_rng(11)
    reference = set()

    def signature(observation, following, action, reward, terminated, truncated):
        return (
            hashlib.sha256(observation.tobytes() + following.tobytes()).digest(),
            int(action),
            float(reward),
            bool(terminated),
            bool(truncated),
        )

    try:
        observation, _ = env.reset(seed=11)
        replay.begin_episode(observation[-1])
        for _ in range(80):
            action = int(rng.integers(env.action_space.n))
            following, reward, terminated, truncated, _ = env.step(action)
            reference.add(signature(observation, following, action, reward, terminated, truncated))
            replay.add(action, reward, following[-1], terminated, truncated)
            batch = replay.sample(8)
            for index in range(8):
                assert (
                    signature(
                        batch.observations[index],
                        batch.next_observations[index],
                        batch.actions[index],
                        batch.rewards[index],
                        batch.terminated[index],
                        batch.truncated[index],
                    )
                    in reference
                )
            observation = following
            if terminated or truncated:
                observation, _ = env.reset()
                replay.begin_episode(observation[-1])
        assert len(reference) > 20  # Exercise changing game images, not identical reset frames.
    finally:
        env.close()
