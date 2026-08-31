"""MimicGen keypose target adapter for attention supervision."""

import numpy as np

from visuomotor.data.core import keypose_targets as CoreKeyposeTargets
from visuomotor.data.core import segmentation as CoreSegmentation


class MimicGenKeyposeTargetAdapter:
    """Derive policy target fields without creating a policy-specific dataset."""

    def __init__(
        self,
        *,
        first_indices: np.ndarray,
        last_indices: np.ndarray,
        valid: np.ndarray,
        command_keyframes_per_episode=None,
    ) -> None:
        self.first_indices = first_indices
        self.last_indices = last_indices
        self.valid = valid
        self.command_keyframes_per_episode = tuple(
            np.asarray(indices, dtype=np.int64)
            for indices in (command_keyframes_per_episode or ())
        )

    @classmethod
    def from_dataset(
        cls,
        dataset,
        *,
        gripper_motion_threshold: float,
        gripper_valley_threshold: float,
        gripper_valley_window: int,
    ):
        keyframes, command_keyframes = (
            CoreSegmentation.segment_settled_gripper_commands(
                dataset.action[:, -1],
                dataset.lowdim[dataset.gripper_key],
                dataset.cum_lengths_all[1:],
                motion_threshold=float(gripper_motion_threshold),
                valley_threshold=float(gripper_valley_threshold),
                valley_window=int(gripper_valley_window),
            )
        )
        first, last, _, valid = CoreSegmentation.build_keypose_segments_from_keyframes(
            keyframes,
            dataset.cum_lengths_all[1:],
        )
        return cls(
            first_indices=first,
            last_indices=last,
            valid=valid,
            command_keyframes_per_episode=command_keyframes,
        )

    def fields(self, dataset, global_indices: np.ndarray) -> dict:
        """Return reference-frame and focus targets for selected global rows."""
        position = dataset.lowdim[dataset.pos_key]
        last_indices = self.last_indices[global_indices]
        return {
            "focus_target_pos": position[last_indices].astype(np.float32, copy=False),
            "focus_target_valid": self.valid[global_indices],
        }
