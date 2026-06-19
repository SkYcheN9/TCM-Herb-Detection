"""Project constants shared by dataset and training scripts."""

from __future__ import annotations

CLASS_NAMES: list[str] = [
    "zexie",
    "niuxi",
    "gaoliangjiang",
    "mudanpi",
    "yuzhu",
    "baizhi",
    "baishao",
    "dazao",
    "danshen",
    "gancao",
    "baixianpi",
    "baihe",
    "sangzhi",
    "jiegeng",
    "banlangen",
]

IMAGE_SUFFIXES: set[str] = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_RAW_IMAGE_DIR = "data/images"
DEFAULT_RAW_LABEL_DIR = "data/labels"
