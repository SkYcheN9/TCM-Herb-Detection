"""Shared class names and display labels."""

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

CHINESE_CLASS_NAMES: dict[str, str] = {
    "zexie": "\u6cfd\u6cfb",
    "niuxi": "\u725b\u819d",
    "gaoliangjiang": "\u9ad8\u826f\u59dc",
    "mudanpi": "\u7261\u4e39\u76ae",
    "yuzhu": "\u7389\u7af9",
    "baizhi": "\u767d\u82b7",
    "baishao": "\u767d\u828d",
    "dazao": "\u5927\u67a3",
    "danshen": "\u4e39\u53c2",
    "gancao": "\u7518\u8349",
    "baixianpi": "\u767d\u9c9c\u76ae",
    "baihe": "\u767e\u5408",
    "sangzhi": "\u6851\u679d",
    "jiegeng": "\u6854\u6897",
    "banlangen": "\u677f\u84dd\u6839",
}

IMAGE_EXTENSIONS: set[str] = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS: set[str] = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}
