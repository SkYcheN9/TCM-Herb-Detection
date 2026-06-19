"""Project-specific Ultralytics trainer extensions."""

from __future__ import annotations

from ultralytics.models.yolo.detect.train import DetectionTrainer


class FocalDetectionTrainer(DetectionTrainer):
    """Detection trainer that injects project focal-loss settings into the model."""

    enable_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float | None = 0.25

    def set_model_attributes(self) -> None:
        """Attach dataset metadata and focal-loss settings to the model args."""

        super().set_model_attributes()
        self.model.enable_focal_loss = self.enable_focal_loss
        self.model.focal_gamma = self.focal_gamma
        self.model.focal_alpha = self.focal_alpha


def build_focal_trainer(
    enable_focal_loss: bool,
    focal_gamma: float,
    focal_alpha: float | None,
) -> type[DetectionTrainer]:
    """Create a trainer class carrying the selected focal-loss settings."""

    class ConfiguredFocalDetectionTrainer(FocalDetectionTrainer):
        pass

    ConfiguredFocalDetectionTrainer.enable_focal_loss = enable_focal_loss
    ConfiguredFocalDetectionTrainer.focal_gamma = float(focal_gamma)
    ConfiguredFocalDetectionTrainer.focal_alpha = focal_alpha
    return ConfiguredFocalDetectionTrainer
