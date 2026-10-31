"""Frames: entities with slots, defaults, ranges, inheritance and if-needed procedures (Task 1).

A frame is either a class frame (e.g. Business, Move) or an instance frame (e.g. the
founder's shop). Slot lookup order, as in Minsky's frames:
  1. the frame's own value
  2. the frame's own if-needed procedure (computes the value from other slots)
  3. the same two steps on the parent frame (is_a / instance_of), up the chain
  4. the nearest default
Each slot can declare a range; `violations()` lists values outside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Slot:
    value: Any = None
    default: Any = None
    range: tuple[float, float] | tuple[str, ...] | None = None  # numeric (lo, hi) or allowed values
    if_needed: Callable[["Frame"], Any] | None = None
    doc: str = ""


@dataclass
class Frame:
    name: str
    parent: "Frame | None" = None
    slots: dict[str, Slot] = field(default_factory=dict)
    is_class: bool = False

    # ---- definition ----
    def define(self, slot: str, **facets) -> "Frame":
        self.slots[slot] = Slot(**facets)
        return self

    def set(self, slot: str, value: Any) -> "Frame":
        if slot in self.slots:
            self.slots[slot].value = value
        else:
            self.slots[slot] = Slot(value=value)
        return self

    # ---- lookup ----
    def _find(self, slot: str) -> tuple["Frame", Slot] | None:
        f = self
        while f is not None:
            if slot in f.slots:
                return f, f.slots[slot]
            f = f.parent
        return None

    def get(self, slot: str, explain: bool = False):
        """Value of a slot, following own value -> if-needed -> parent chain -> default."""
        f: Frame | None = self
        default, default_from = None, None
        while f is not None:
            s = f.slots.get(slot)
            if s is not None:
                if s.value is not None:
                    return (s.value, f"value on {f.name}") if explain else s.value
                if s.if_needed is not None:
                    v = s.if_needed(self)
                    if v is not None:
                        return (v, f"computed by if-needed on {f.name}") if explain else v
                if s.default is not None and default is None:
                    default, default_from = s.default, f.name
            f = f.parent
        return (default, f"default from {default_from}" if default_from else "unknown") if explain else default

    def isa(self, class_name: str) -> bool:
        f: Frame | None = self
        while f is not None:
            if f.name == class_name:
                return True
            f = f.parent
        return False

    def lineage(self) -> list[str]:
        out, f = [], self
        while f is not None:
            out.append(f.name)
            f = f.parent
        return out

    def all_slots(self) -> list[str]:
        names, f = [], self
        while f is not None:
            names += [s for s in f.slots if s not in names]
            f = f.parent
        return names

    def violations(self) -> list[str]:
        """Slot values outside their declared ranges (the range may be inherited)."""
        out = []
        for slot in self.all_slots():
            v = self.get(slot)
            rng = None
            f = self
            while f is not None and rng is None:  # nearest declared range
                if slot in f.slots and f.slots[slot].range is not None:
                    rng = f.slots[slot].range
                f = f.parent
            if v is None or rng is None:
                continue
            if len(rng) == 2 and all(isinstance(x, (int, float)) for x in rng):
                if not isinstance(v, (int, float)) or not (rng[0] <= v <= rng[1]):
                    out.append(f"{self.name}.{slot} = {v!r} is outside {rng}")
            elif v not in rng:
                out.append(f"{self.name}.{slot} = {v!r} is not one of {rng}")
        return out

    def describe(self) -> dict[str, Any]:
        return {s: self.get(s) for s in self.all_slots()}

    def instance(self, name: str, **values) -> "Frame":
        inst = Frame(name=name, parent=self)
        for k, v in values.items():
            inst.set(k, v)
        return inst
