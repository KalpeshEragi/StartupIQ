"""Uninformed and informed search over plans (Task 3, decision 0008).

All five share one loop shape and count the same things: nodes expanded (successors generated
from them), nodes generated, frontier peak, time. They differ only in the frontier:

    BFS     FIFO queue, goal test on generation -> fewest moves, not cheapest
    DFS     LIFO stack, depth-limited          -> first plan found, any cost
    UCS     priority g (cost so far)           -> cheapest plan (optimal reference)
    A*      priority g + h                     -> cheapest if h is admissible, fewer expansions
    Greedy  priority h                         -> fast, no optimality guarantee
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass

from startupiq.search.problem import Problem

NODE_LIMIT = 150_000


@dataclass
class SearchResult:
    algorithm: str
    plan: tuple | None
    cost: float | None
    expanded: int
    generated: int
    frontier_peak: int
    ms: float
    hit_limit: bool = False


def _done(name, plan, cost, expanded, generated, peak, t0, hit=False) -> SearchResult:
    return SearchResult(name, plan, cost, expanded, generated, peak, (time.perf_counter() - t0) * 1000, hit)


def bfs(p: Problem) -> SearchResult:
    t0 = time.perf_counter()
    if p.is_goal(()):
        return _done("BFS", (), 0.0, 0, 1, 1, t0)
    q, seen_cost = deque([((), 0.0)]), None
    expanded = generated = peak = 0
    while q:
        plan, g = q.popleft()
        expanded += 1
        for nxt, c in p.successors(plan):
            generated += 1
            if p.is_goal(nxt):
                return _done("BFS", nxt, g + c, expanded, generated, peak, t0)
            q.append((nxt, g + c))
        peak = max(peak, len(q))
        if generated > NODE_LIMIT:
            return _done("BFS", None, seen_cost, expanded, generated, peak, t0, True)
    return _done("BFS", None, None, expanded, generated, peak, t0)


def dfs(p: Problem) -> SearchResult:
    t0 = time.perf_counter()
    stack = [((), 0.0)]
    expanded = generated = peak = 0
    while stack:
        plan, g = stack.pop()
        if p.is_goal(plan):
            return _done("DFS", plan, g, expanded, generated, peak, t0)
        expanded += 1
        for nxt, c in p.successors(plan):  # depth limit = MAX_MOVES, enforced by successors()
            generated += 1
            stack.append((nxt, g + c))
        peak = max(peak, len(stack))
        if generated > NODE_LIMIT:
            return _done("DFS", None, None, expanded, generated, peak, t0, True)
    return _done("DFS", None, None, expanded, generated, peak, t0)


def _best_first(p: Problem, name: str, priority) -> SearchResult:
    t0 = time.perf_counter()
    tie = itertools.count()
    frontier = [(priority((), 0.0), next(tie), (), 0.0)]
    best_g: dict[tuple, float] = {(): 0.0}
    expanded = generated = peak = 0
    while frontier:
        _, _, plan, g = heapq.heappop(frontier)
        if g > best_g.get(plan, float("inf")):
            continue
        if p.is_goal(plan):  # goal test on expansion keeps UCS and A* optimal
            return _done(name, plan, g, expanded, generated, peak, t0)
        expanded += 1
        for nxt, c in p.successors(plan):
            generated += 1
            g2 = g + c
            if g2 < best_g.get(nxt, float("inf")):
                best_g[nxt] = g2
                heapq.heappush(frontier, (priority(nxt, g2), next(tie), nxt, g2))
        peak = max(peak, len(frontier))
        if generated > NODE_LIMIT:
            return _done(name, None, None, expanded, generated, peak, t0, True)
    return _done(name, None, None, expanded, generated, peak, t0)


def ucs(p: Problem) -> SearchResult:
    return _best_first(p, "Uniform-cost", lambda plan, g: g)


def astar(p: Problem) -> SearchResult:
    return _best_first(p, "A*", lambda plan, g: g + p.heuristic(plan))


def greedy(p: Problem) -> SearchResult:
    return _best_first(p, "Greedy best-first", lambda plan, g: p.heuristic(plan))


ALGORITHMS = {"bfs": bfs, "dfs": dfs, "ucs": ucs, "astar": astar, "greedy": greedy}
