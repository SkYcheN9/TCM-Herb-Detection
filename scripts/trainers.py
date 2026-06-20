"""Project-specific Ultralytics trainer extensions."""

from __future__ import annotations

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


class ProjectDetectionTrainer(DetectionTrainer):
    """Detection trainer that injects project settings into the model."""

    enable_focal_loss: bool = False
    focal_loss_type: str = "soft_focal"
    focal_gamma: float = 1.0
    focal_alpha: float | None = None
    pretrained_transfer: bool = False
    pretrained_weights: str = "yolov8n.pt"

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO model, optionally using safe partial pretrained transfer."""

        if not self.pretrained_transfer:
            return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        from scripts.pretrained import transfer_pretrained_weights

        report = transfer_pretrained_weights(model, self.pretrained_weights)
        model.pretrained_transfer_report = report
        LOGGER.info(report.summary())
        return model

    def set_model_attributes(self) -> None:
        """Attach dataset metadata and focal-loss settings to the model args."""

        super().set_model_attributes()
        self.model.enable_focal_loss = self.enable_focal_loss
        self.model.focal_loss_type = self.focal_loss_type
        self.model.focal_gamma = self.focal_gamma
        self.model.focal_alpha = self.focal_alpha
        self.model.pretrained_transfer = self.pretrained_transfer
        self.model.pretrained_weights = self.pretrained_weights


def build_project_trainer(
    enable_focal_loss: bool,
    focal_gamma: float,
    focal_alpha: float | None,
    focal_loss_type: str = "soft_focal",
    pretrained_transfer: bool = False,
    pretrained_weights: str = "yolov8n.pt",
) -> type[DetectionTrainer]:
    """Create a trainer class carrying selected project settings."""

    class ConfiguredProjectDetectionTrainer(ProjectDetectionTrainer):
        pass

    ConfiguredProjectDetectionTrainer.enable_focal_loss = enable_focal_loss
    ConfiguredProjectDetectionTrainer.focal_loss_type = focal_loss_type
    ConfiguredProjectDetectionTrainer.focal_gamma = float(focal_gamma)
    ConfiguredProjectDetectionTrainer.focal_alpha = focal_alpha
    ConfiguredProjectDetectionTrainer.pretrained_transfer = pretrained_transfer
    ConfiguredProjectDetectionTrainer.pretrained_weights = pretrained_weights
    return ConfiguredProjectDetectionTrainer


def build_focal_trainer(
    enable_focal_loss: bool,
    focal_gamma: float,
    focal_alpha: float | None,
    focal_loss_type: str = "soft_focal",
) -> type[DetectionTrainer]:
    """Create a focal-loss trainer; kept for backward-compatible imports."""

    return build_project_trainer(
        enable_focal_loss=enable_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        focal_loss_type=focal_loss_type,
    )
