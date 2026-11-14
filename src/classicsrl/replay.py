"""Uniform replay with one frame arena and explicit episode boundaries.

Each transition points to current and next frames in the arena. Stacks are rebuilt
when sampled. Old transitions are evicted as soon as a required frame is overwritten.
All frame IDs are absolute, so ring wraparound never joins unrelated observations.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, frame_shape=(84, 84), stack_size=4, seed=0):
        if capacity <= 0 or stack_size <= 0:
            raise ValueError("capacity and stack_size must be positive")
        self.capacity = capacity
        self.stack_size = stack_size
        self.frame_shape = tuple(frame_shape)
        self.frames = np.zeros((capacity + stack_size, *frame_shape), dtype=np.uint8)
        self.current_ids = np.zeros(capacity, dtype=np.int64)
        self.next_ids = np.zeros(capacity, dtype=np.int64)
        self.episode_starts = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.frame_count = 0
        self.transition_count = 0
        self.oldest_transition = 0
        self.current_frame = None
        self.episode_start = None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.transition_count - self.oldest_transition

    @property
    def nbytes(self):
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def _write_frame(self, frame):
        frame = np.asarray(frame)
        if frame.shape != self.frame_shape or frame.dtype != np.uint8:
            raise ValueError(f"frames must be uint8 with shape {self.frame_shape}")
        frame_id = self.frame_count
        self.frames[frame_id % len(self.frames)] = frame
        self.frame_count += 1
        return frame_id

    def _evict(self):
        oldest_frame = self.frame_count - len(self.frames)
        self.oldest_transition = max(self.oldest_transition, self.transition_count - self.capacity)
        while self.oldest_transition < self.transition_count:
            slot = self.oldest_transition % self.capacity
            first_needed = max(
                self.episode_starts[slot], self.current_ids[slot] - self.stack_size + 1
            )
            if first_needed >= oldest_frame:
                break
            self.oldest_transition += 1

    def begin_episode(self, initial_frame):
        """Start a new frame stack, repeating the reset frame as Gymnasium does."""
        self.current_frame = self._write_frame(initial_frame)
        self.episode_start = self.current_frame
        self._evict()

    def add(self, action, reward, next_frame, terminated, truncated):
        if self.current_frame is None:
            raise ValueError("call begin_episode before adding transitions")
        next_id = self._write_frame(next_frame)
        slot = self.transition_count % self.capacity
        self.current_ids[slot] = self.current_frame
        self.next_ids[slot] = next_id
        self.episode_starts[slot] = self.episode_start
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.terminated[slot] = terminated
        self.truncated[slot] = truncated
        self.transition_count += 1
        self.current_frame = None if terminated or truncated else next_id
        self._evict()

    def _stacks(self, frame_ids, starts):
        offsets = np.arange(1 - self.stack_size, 1)
        ids = np.maximum(frame_ids[:, None] + offsets, starts[:, None])
        return self.frames[ids % len(self.frames)].copy()

    def sample(self, batch_size: int, *, rng=None) -> Batch:
        if not len(self):
            raise ValueError("cannot sample an empty buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        generator = self.rng if rng is None else rng
        slots = (
            generator.integers(self.oldest_transition, self.transition_count, size=batch_size)
            % self.capacity
        )
        starts = self.episode_starts[slots]
        return Batch(
            self._stacks(self.current_ids[slots], starts),
            self.actions[slots].copy(),
            self.rewards[slots].copy(),
            self._stacks(self.next_ids[slots], starts),
            self.terminated[slots].copy(),
            self.truncated[slots].copy(),
        )

    def save(self, path: Path):
        metadata = {
            "schema_version": 1,
            "capacity": self.capacity,
            "frame_shape": self.frame_shape,
            "stack_size": self.stack_size,
            "frame_count": self.frame_count,
            "transition_count": self.transition_count,
            "oldest_transition": self.oldest_transition,
            "rng": self.rng.bit_generator.state,
        }
        arrays = {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}
        np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["schema_version"] != 1:
                raise ValueError("unsupported replay schema")
            replay = cls(meta["capacity"], meta["frame_shape"], meta["stack_size"])
            for name, value in vars(replay).items():
                if isinstance(value, np.ndarray):
                    stored = archive[name]
                    if stored.shape != value.shape or stored.dtype != value.dtype:
                        raise ValueError(f"invalid replay array: {name}")
                    value[:] = stored
        for name in ("frame_count", "transition_count", "oldest_transition"):
            setattr(replay, name, meta[name])
        replay.rng.bit_generator.state = meta["rng"]
        # Deliberately resume at a fresh episode; no emulator state is restored.
        return replay
