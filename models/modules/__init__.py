"""Custom neural network modules used by project model YAML files."""

from .bifpn import BiFPNFusion
from .bifpn import register_ultralytics_modules as register_bifpn_modules
from .cbam import CBAM
from .cbam import register_ultralytics_modules as register_cbam_modules
from .decoupled_head import DecoupledDetect
from .decoupled_head import register_ultralytics_modules as register_decoupled_head_modules


def register_ultralytics_modules(
    enable_cbam: bool = True,
    enable_bifpn: bool = False,
    enable_decoupled_head: bool = False,
) -> None:
    """Register project modules needed by the selected model YAML."""

    if enable_cbam:
        register_cbam_modules()
    if enable_bifpn:
        register_bifpn_modules()
    if enable_decoupled_head:
        register_decoupled_head_modules()


__all__ = ["BiFPNFusion", "CBAM", "DecoupledDetect", "register_ultralytics_modules"]
