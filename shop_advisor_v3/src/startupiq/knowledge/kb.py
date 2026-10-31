"""Knowledge base = class frames + instance frames + a semantic network of relations (Task 1).

Relations (edge labels in the network):
    is_a            class -> parent class
    instance_of     instance -> class
    operates_in     business -> category
    sells           business -> product
    belongs_to      product -> category
    competes_with   business -> competitor (both directions)
    affects         move class -> metric (edge attribute sign: + or -)
    requires        move class -> constraint
"""

from __future__ import annotations

import networkx as nx

from startupiq.knowledge.constraints import CONSTRAINTS
from startupiq.knowledge.frames import Frame
from startupiq.knowledge.ontology import build_classes, move_names

METRICS = ["revenue", "margin", "share", "cash", "runway", "stock_cover", "lost_sales", "retention", "dependence"]


class KnowledgeBase:
    def __init__(self) -> None:
        self.classes = build_classes()
        self.instances: dict[str, Frame] = {}
        self.net = nx.MultiDiGraph()
        for name, f in self.classes.items():
            self.net.add_node(name, kind="class")
            if f.parent is not None:
                self.net.add_edge(name, f.parent.name, label="is_a")
        for m in METRICS:
            self.net.add_node(m, kind="metric")
            self.net.add_edge(m, "Metric", label="instance_of")
        for c in CONSTRAINTS.values():
            self.net.add_node(c.name, kind="constraint", statement=c.statement)
            self.net.add_edge(c.name, "Constraint", label="instance_of")
        for mv in move_names(self.classes):
            f = self.classes[mv]
            for metric, sign in (f.get("affects") or {}).items():
                self.net.add_edge(mv, metric, label="affects", sign=sign)
            for req in f.get("requires") or []:
                self.net.add_edge(mv, req, label="requires")

    # ---- building ----
    def add(self, cls: str, name: str, **slots) -> Frame:
        inst = self.classes[cls].instance(name, **{k: v for k, v in slots.items() if v is not None})
        self.instances[name] = inst
        self.net.add_node(name, kind="instance")
        self.net.add_edge(name, cls, label="instance_of")
        return inst

    def relate(self, a: str, label: str, b: str, **attrs) -> None:
        self.net.add_edge(a, b, label=label, **attrs)
        if label == "competes_with":
            self.net.add_edge(b, a, label=label, **attrs)

    # ---- queries ----
    def related(self, node: str, label: str) -> list[str]:
        return [v for _, v, d in self.net.out_edges(node, data=True) if d["label"] == label]

    def moves_affecting(self, metric: str, sign: str = "+") -> list[str]:
        return sorted(u for u, _, d in self.net.in_edges(metric, data=True) if d["label"] == "affects" and d.get("sign") == sign)

    def explain_path(self, a: str, b: str) -> list[str]:
        """Shortest chain of relations from a to b, as readable steps (used in explanations)."""
        path = nx.shortest_path(self.net.to_undirected(as_view=True), a, b)
        steps = []
        for u, v in zip(path, path[1:]):
            d = self.net.get_edge_data(u, v) or self.net.get_edge_data(v, u)
            label = next(iter(d.values()))["label"]
            steps.append(f"{u} --{label}--> {v}" if self.net.has_edge(u, v) else f"{v} --{label}--> {u}")
        return steps

    # ---- consistency ----
    def check(self, business: str, moves: list[str] | None = None) -> list[dict]:
        """Violations for the business itself and for each proposed move."""
        biz = self.instances[business]
        out = [{"subject": business, "constraint": "slot_range", "reason": r} for r in biz.violations()]
        for comp in self.related(business, "competes_with"):
            mine, theirs = self.related(business, "operates_in"), self.related(comp, "operates_in")
            if mine and theirs and not set(mine) & set(theirs):
                out.append({"subject": comp, "constraint": "competitor_same_category",
                            "reason": f"{comp} operates in {theirs}, not in {mine}"})
        for mv in moves or []:
            if mv not in self.classes:
                out.append({"subject": mv, "constraint": "known_move", "reason": "not a move in the catalogue"})
                continue
            move = self.classes[mv]
            for req in move.get("requires") or []:
                reason = CONSTRAINTS[req].test(biz, move)
                if reason:
                    out.append({"subject": mv, "constraint": req, "reason": reason})
        return out

    def allowed_moves(self, business: str) -> list[str]:
        """Moves whose own constraints all hold for this business."""
        base = len(self.check(business))
        return [m for m in move_names(self.classes) if len(self.check(business, [m])) == base]
