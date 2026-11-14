import numpy as np
import pytest

from classicsrl.replay import ReplayBuffer


@pytest.mark.parametrize("capacity", [1, 7, 13])
def test_reconstructed_stacks_match_explicit_reference_across_resets_and_wraparound(capacity):
    replay = ReplayBuffer(capacity, frame_shape=(2, 3), seed=23)
    rng = np.random.default_rng(1)
    reference = {}
    action = 0
    frame_number = 0
    for episode in range(30):
        initial = np.full((2, 3), frame_number % 256, np.uint8)
        frame_number += 1
        observation = np.repeat(initial[None], 4, axis=0)
        replay.begin_episode(initial)
        length = int(rng.integers(1, 10))
        for step in range(length):
            frame = np.full((2, 3), frame_number % 256, np.uint8)
            frame_number += 1
            next_observation = np.concatenate((observation[1:], frame[None]))
            ended = step == length - 1
            terminated = ended and episode % 2 == 0
            truncated = ended and not terminated
            replay.add(action, action + 0.5, frame, terminated, truncated)
            reference[action] = (observation.copy(), next_observation.copy(), terminated, truncated)
            assert 0 < len(replay) <= capacity
            batch = replay.sample(32)
            for index, sampled_action in enumerate(batch.actions):
                current, following, terminal, timeout = reference[sampled_action]
                np.testing.assert_array_equal(batch.observations[index], current)
                np.testing.assert_array_equal(batch.next_observations[index], following)
                assert batch.rewards[index] == sampled_action + 0.5
                assert batch.terminated[index] == terminal
                assert batch.truncated[index] == timeout
                assert sampled_action > action - capacity
            observation = next_observation
            action += 1


def test_terminal_successor_is_not_replaced_by_reset_observation():
    replay = ReplayBuffer(10, frame_shape=(1, 1))
    replay.begin_episode(np.array([[10]], np.uint8))
    replay.add(0, 1, np.array([[11]], np.uint8), False, True)
    replay.begin_episode(np.array([[99]], np.uint8))
    batch = replay.sample(1)
    assert batch.next_observations[0, :, 0, 0].tolist() == [10, 10, 10, 11]
    assert not batch.terminated[0] and batch.truncated[0]


def test_restore_preserves_sampling_rng_and_starts_new_episode(tmp_path):
    replay = ReplayBuffer(8, frame_shape=(2, 2), seed=99)
    replay.begin_episode(np.zeros((2, 2), np.uint8))
    for i in range(20):
        replay.add(i, i, np.full((2, 2), i + 1, np.uint8), False, False)
    path = tmp_path / "replay.npz"
    replay.save(path)
    restored = ReplayBuffer.load(path)
    expected, actual = replay.sample(16), restored.sample(16)
    for name in vars(expected):
        np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))
    with pytest.raises(ValueError, match="begin_episode"):
        restored.add(0, 0, np.zeros((2, 2), np.uint8), False, False)
    restored.begin_episode(np.full((2, 2), 100, np.uint8))
    restored.add(100, 0, np.full((2, 2), 101, np.uint8), False, False)
    sample = restored.sample(100)
    new = sample.observations[sample.actions == 100]
    assert len(new) > 0
    assert (new == 100).all()


def test_byte_storage_uses_one_frame_arena():
    replay = ReplayBuffer(100, frame_shape=(84, 84))
    assert replay.frames.nbytes == 104 * 84 * 84
    assert replay.nbytes < 2 * 100 * 84 * 84


def test_sampling_empty_or_adding_after_terminal_is_rejected():
    replay = ReplayBuffer(5, frame_shape=(1, 1))
    with pytest.raises(ValueError, match="empty"):
        replay.sample(1)
    replay.begin_episode(np.zeros((1, 1), np.uint8))
    replay.add(0, 0, np.zeros((1, 1), np.uint8), True, False)
    with pytest.raises(ValueError, match="begin_episode"):
        replay.add(0, 0, np.zeros((1, 1), np.uint8), False, False)
