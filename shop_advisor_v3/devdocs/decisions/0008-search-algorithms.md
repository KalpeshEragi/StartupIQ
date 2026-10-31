# 0008: Search algorithms for the planner

- Status: Accepted (2026-10-24); implemented formulation and results in decision 0014
- Date: 2026-10-24
- Module / task: M9; Task 3

## State space

- State: (week, cash, price level, active promotion, stock units, moves taken).
- Actions: hold, order stock, run promotion, change price by a step, add channel, cut cost. Each action has a cash cost and preconditions from the knowledge model.
- Transition: the forecaster predicts demand given price and promotion; stock and cash update by arithmetic.
- Goal: reach a revenue or share target by week H, at least cost. A second variant maximises projected profit.

## Algorithms and why each is included

| Algorithm | Why | Expected behaviour |
| --- | --- | --- |
| BFS | Uninformed; optimal in number of moves | Finds the shortest plan, not the cheapest; explodes in node count |
| DFS (depth-limited) | Uninformed; low memory | Fast first answer, often poor |
| Uniform-cost search | Optimal on cost, no heuristic | Reference for checking A* |
| **A\*** | Optimal on cost if the heuristic is admissible; fewer expansions | Main planner |
| Greedy best-first | Uses heuristic only | Fast, not optimal |

## Heuristic

h(state) = remaining revenue gap divided by the best revenue gain per unit cost of any legal action, using the forecaster's optimistic (P90) response. This never overestimates the true cost only if the P90 response really is an upper bound, which a learned model cannot guarantee. We therefore check A\* against uniform-cost search on short horizons and report any case where they disagree.

## Comparison to report

Plan cost, plan value, nodes expanded, time, and optimality gap vs uniform-cost search, on 3 to 4 business scenarios.
