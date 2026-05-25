"""持久層：把記憶存成 JSON 檔，重啟後讀得回來，並用 id 去重。
load()/_persist()/add() 要你填。"""
from __future__ import annotations
import json
import os


class JsonStore:
    def __init__(self, path: str):
        self.path = path
        self.items: list[dict] = []
        self.load()

    def load(self) -> None:
        """TODO：從硬碟讀回 self.items。
        檔案不存在、解析失敗、或內容不是 list（例如 {}）都一律視為空 list。
        提示：os.path.exists、open(...,encoding='utf-8')、json.load、isinstance(data, list)。"""
        # TODO: 換成真的讀檔。先給空 list 讓建構不會壞。
        self.items = []

    def _persist(self) -> None:
        """TODO：把 self.items 寫回硬碟（JSON）。
        提示：json.dump(self.items, f, ensure_ascii=False, indent=2)。"""
        raise NotImplementedError("JsonStore._persist not implemented yet")

    def add(self, obs: dict) -> bool:
        """TODO：新增一筆；id 已存在則跳過，回傳是否真的新增。新增後要 _persist()。
        提示：any(o['id'] == obs['id'] for o in self.items)。"""
        raise NotImplementedError("JsonStore.add not implemented yet")

    def all(self) -> list[dict]:
        return list(self.items)

    def clear(self) -> None:
        self.items = []
        self._persist()
