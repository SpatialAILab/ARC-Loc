import torch
import torch.nn as nn


class ARCCorrespondenceMatcher(nn.Module):
    """Compute ARC-Loc correspondence probabilities from descriptor similarities."""

    def __init__(self, initial_dustbin_score=1.0):
        super().__init__()

        self.unmatched_score = nn.Parameter(
            torch.tensor(float(initial_dustbin_score))
        )

    def forward(self, similarity):
        """
        Args:
            similarity:
                Pairwise descriptor similarity with shape [B, M, N].

        Returns:
            Matching probabilities with shape [B, M, N].
        """
        batch_size, num_satellite, num_ground = similarity.shape

        augmented = similarity.new_empty(
            batch_size,
            num_satellite + 1,
            num_ground + 1,
        )

        # Descriptor-to-descriptor similarities.
        augmented[
            :, :num_satellite, :num_ground
        ] = similarity

        # The final row and column represent unmatched features.
        augmented[
            :, :num_satellite, num_ground
        ] = self.unmatched_score

        augmented[
            :, num_satellite, :num_ground
        ] = self.unmatched_score

        augmented[
            :, num_satellite, num_ground
        ] = self.unmatched_score

        satellite_conditioned = torch.softmax(
            augmented,
            dim=2,
        )

        ground_conditioned = torch.softmax(
            augmented,
            dim=1,
        )

        mutual_probability = (
            satellite_conditioned
            * ground_conditioned
        )

        return mutual_probability[
            :, :num_satellite, :num_ground
        ]