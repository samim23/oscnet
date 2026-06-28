"""Per-class positive-memory queue for conditional generator losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .common import Array

@dataclass
class MNISTDriftQueue:
    """Host-side per-class FIFO memory for conditional drift positives."""

    images: np.ndarray
    counts: np.ndarray
    write_indices: np.ndarray
    num_classes: int
    queue_size: int
    image_dim: int
    rng: np.random.Generator

    @classmethod
    def create(
        cls,
        *,
        num_classes: int,
        queue_size: int,
        image_dim: int,
        seed: int,
    ) -> "MNISTDriftQueue":
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if queue_size < 1:
            raise ValueError("drift_queue_size must be positive")
        return cls(
            images=np.zeros(
                (int(num_classes), int(queue_size), int(image_dim)),
                dtype=np.float32,
            ),
            counts=np.zeros((int(num_classes),), dtype=np.int32),
            write_indices=np.zeros((int(num_classes),), dtype=np.int32),
            num_classes=int(num_classes),
            queue_size=int(queue_size),
            image_dim=int(image_dim),
            rng=np.random.default_rng(int(seed)),
        )

    def push(self, images: Array, labels: Array) -> None:
        images_np = np.asarray(images, dtype=np.float32).reshape(-1, self.image_dim)
        labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
        for image, label in zip(images_np, labels_np):
            class_id = int(label)
            if class_id < 0 or class_id >= self.num_classes:
                continue
            slot = int(self.write_indices[class_id])
            self.images[class_id, slot] = image
            self.write_indices[class_id] = (slot + 1) % self.queue_size
            self.counts[class_id] = min(
                int(self.counts[class_id]) + 1,
                self.queue_size,
            )

    def ready(self, num_pos: int) -> bool:
        if num_pos < 1:
            raise ValueError("drift_queue_num_pos must be positive")
        return bool(np.all(self.counts >= int(num_pos)))

    def draw(self, num_pos: int) -> Tuple[Array, Array]:
        if not self.ready(num_pos):
            raise ValueError("drift queue is not ready for the requested positives")
        images = []
        labels = []
        for class_id in range(self.num_classes):
            available = int(self.counts[class_id])
            indices = self.rng.choice(available, size=int(num_pos), replace=False)
            images.append(self.images[class_id, indices])
            labels.append(np.full((int(num_pos),), class_id, dtype=np.int32))
        return jnp.asarray(np.concatenate(images, axis=0)), jnp.asarray(
            np.concatenate(labels, axis=0)
        )


