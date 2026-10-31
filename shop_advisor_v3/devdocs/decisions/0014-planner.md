# 0014: Planner (state-space search over business moves)

- Status: Accepted (2026-10-24). Refines decision 0008 with the implemented formulation and results.
- Module / task: M9; Task 3
- Code: `src/startupiq/search/` (problem, algorithms, plan, `__main__` for the comparison); results `docs/results/planner.md` (`python -m startupiq.search`)

## Formulation

| Element | Choice | Why |
| --- | --- | --- |
| State | A plan: sorted (start week, move) pairs, up to 4 moves, start weeks 0, 2, ..., 10 | Small enough to search exhaustively (uniform-cost is the optimal reference), rich enough to need search |
| Actions | Add a move at or after the plan's last start week (canonical order: each set generated once) if the knowledge model's constraints allow it at that point (cash then, stock then, margin floor, runway, legality) and the rule engine has not forbidden it | Reuses Task 1 and Task 2 instead of duplicating them |
| Transition | A 12-week simulator on the Auto forecast: demand lift and margin change while a move is active, routine weekly reorders, extra stock from a restock after the supplier lead time, fixed costs, cash | The forecast is the "world model" of decision 0008 |
| Goal | 12-week gross profit >= target and cash never below zero | The founder's question: what is the cheapest way to make X more? |
| Cost | Cash spent on the move (stock: 5% holding cost, because the cash returns as sales) + margin given away by a price cut | What the founder gives up, so every move except hold has a positive cost |
| Default target | Doing nothing + 3% of 12-week sales | Modest and visible; the founder can set another |
| Heuristic | Fractional-knapsack bound: cover the remaining profit gap with the remaining moves, best profit per unit cost first, fractions allowed | Lower bound on the remaining cost if moves do not gain more together than apart; checked against uniform-cost search |

## Two modelling mistakes found while testing, both fixed before the comparison

1. **No routine reorders.** The first simulator never restocked unless the plan said so, so doing nothing sold the shelves empty (Rossmann store 1 lost 73% of its sales) and every target looked unreachable. Real shops reorder what they sell; the simulator now does, and stock-outs come only from demand above normal or a thin starting buffer.
2. **The heuristic bounded the wrong simulator.** It measured gaps at P90 demand while the goal test uses P50, so at P90 almost every plan already met the target, h was 0 and A* degenerated into uniform-cost search (and was slower). It now uses the same simulator as the goal test.

Also changed: restocking was costed at its full cash outlay; cash spent on stock comes back as sales, so its cost is now the holding cost (5%, an assumption). The cash constraint still checks the order is affordable.

## Results (4 businesses from real uploads x 3 targets; targets 0.5x / 1.5x / 2.5x the best single move's gain)

| Algorithm | Problems solved (of 10 solvable) | Optimal | Mean nodes expanded | Note |
| --- | --- | --- | --- | --- |
| BFS | 10 | 3 | 436 | Fewest moves, often the most expensive (a new product before a cost cut) |
| DFS (depth 4) | 10 | 2 | 3,874 | First plan found; up to 30x the optimal cost |
| Uniform-cost | 10 | 10 | 3,176 | Optimal reference |
| **A\*** | 10 | **10** | **2,835** | Always optimal (0 of 12 mismatches with uniform-cost) |
| Greedy best-first | 10 | 2 | 2,520 | Fast on easy problems, wrong cost in 8 of 10 |

Two problems (tea stall and Olist seller at 2.5x) have no plan: every algorithm exhausted all plans of up to 4 moves, which proves no such plan exists within the cash available.

A* saves the most on medium targets (Olist 159 vs 868 nodes; Rossmann 59 vs 463; UCI 9 vs 453). On the hardest targets it barely helps (Rossmann 14,793 vs 15,716): the only plans that work use 4 moves including an expensive new product, which the relaxed bound cannot anticipate. A tighter bound (e.g. one that respects the 4-move cap per move type more precisely) is future work.

## In the product

`search/plan.py` runs A* when the upload is weekly and the founder gave cash, fixed costs and margin. Moves forbidden by the rule engine or blocked by the safety gate are excluded. The plan appears on the next-move card with its cost, projected 12-week profit, lowest cash point and a note that move effects are assumptions.

## Limits

- Move effects are assumptions (`configs/moves.yaml`); only the promotion lift is grounded in data.
- Deterministic: the plan is optimal for the P50 forecast, not robust to the range. A risk-aware variant (plan against P10) is future work.
- The 4 benchmark businesses use stated, assumed finances, because no dataset has them.
