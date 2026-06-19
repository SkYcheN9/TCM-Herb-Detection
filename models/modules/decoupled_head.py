"""Decoupled detection head compatible with Ultralytics YOLOv8 parsing."""

from __future__ import annotations

import copy

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect

from .parser_patch import patch_parse_model


class DecoupledDetect(Detect):
    """YOLO detection head with explicitly separated regression and classification towers.

    The standard Ultralytics Detect head is already organized into box and class
    branches. This project head makes the separation explicit for the practice
    requirement: each scale owns an independent stem, regression tower and
    classification tower, so classification gradients do not share the same
    convolutional tower with bbox regression.
    """

    def __init__(
        self,
        nc: int = 80,
        head_channels: int | None = None,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        if not ch:
            raise ValueError("DecoupledDetect requires input channels from model parser")

        tower_channels = int(head_channels or max(64, min(256, ch[0])))
        box_channels = max(16, tower_channels // 2, self.reg_max * 4)
        cls_channels = max(tower_channels, min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(in_channels, tower_channels, 1),
                Conv(tower_channels, box_channels, 3),
                Conv(box_channels, box_channels, 3),
                nn.Conv2d(box_channels, 4 * self.reg_max, 1),
            )
            for in_channels in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(in_channels, tower_channels, 1),
                DWConv(tower_channels, tower_channels, 3),
                Conv(tower_channels, cls_channels, 1),
                DWConv(cls_channels, cls_channels, 3),
                Conv(cls_channels, cls_channels, 1),
                nn.Conv2d(cls_channels, self.nc, 1),
            )
            for in_channels in ch
        )

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.end2end = True


def register_ultralytics_modules() -> None:
    """Expose DecoupledDetect and patch Ultralytics parser rules."""

    import ultralytics.nn.tasks as tasks

    tasks.DecoupledDetect = DecoupledDetect

    def transform(source: str) -> str:
        if "                DecoupledDetect,\n" not in source:
            detect_anchor = "                Detect,\n                WorldDetect,"
            if detect_anchor not in source:
                raise RuntimeError("Unable to patch Ultralytics parser for DecoupledDetect")
            source = source.replace(
                detect_anchor,
                "                Detect,\n                DecoupledDetect,\n                WorldDetect,",
            )
        legacy_anchor = (
            "            if m in {Detect, YOLOEDetect, Segment, Segment26, "
            "YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:"
        )
        if legacy_anchor in source:
            source = source.replace(
                legacy_anchor,
                "            if m in {Detect, DecoupledDetect, YOLOEDetect, Segment, "
                "Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:",
            )
        return source

    patch_parse_model("DecoupledDetect", transform)
