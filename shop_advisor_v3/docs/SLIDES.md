# Startup IQ: presentation outline (12 slides, about 12 minutes)

1. **The problem.** Small sellers decide stock, prices and promotions by gut feeling; an analyst costs more than they earn. Startup IQ: upload a sales sheet, get a weekly checkup with reasons.
2. **Live demo.** Upload a file in the web app; walk through the seven cards (data, forecast, money, warnings, market, rivals, next move).
3. **Architecture.** DL for representation (TCN), ML for prediction (Auto, random forest, LightGBM), AI for decisions (rules, constraints, A*). One technique per module, argued in devdocs/decisions/0001.
4. **Data (Task 4).** Four sources, scripted cleaning, intake that reads messy founder files: 25/25 broken-file cases. Figure: 01_rossmann_weekly_total.png.
5. **Knowledge model (Task 1).** Frames with inheritance and if-needed slots; semantic network; constraints. Figures 16-17. 8/8 validation scenarios.
6. **Rule engine (Task 2).** 47 rules, forward and backward chaining, certainty factors from measured model quality. Show one "why not expand?" explanation.
7. **Planner (Task 3).** Plans of moves, simulator on the forecast; A* optimal in 10/10 with 5-50x fewer nodes than uniform-cost on medium targets; BFS/DFS/greedy compared.
8. **Classical ML (Tasks 5-6).** Leak-free features (tested), test scored once. Table: random forest 0.130 vs 0.160 baseline; competitor F1 0.366 vs 0.228; share 1.55 vs 1.99. Market pulse model lost to a rule, so the rule ships.
9. **Deep learning (Task 7).** MLP / LSTM+attention / TCN on GPU. None beats the best classical model alone; all win on sparse products; the hybrid (Auto + TCN) serves. The weekday bug story (0.549 -> 0.145).
10. **Interpretability (Task 8).** Saliency figure 18; attention vs occlusion (rho -0.11: attention is not a faithful explanation); filters fire on spikes (figure 20).
11. **Validation and ethics (Task 10).** 300 unseen sellers: beats naive, not a moving average; ranges too narrow (56%); better for small sellers. Safety gate: illegal moves refused, estimates labelled, thin data refused, uploads not stored.
12. **Limits and next steps.** Pilot with 3-5 real Indian small businesses, calibrate ranges, local category data, licensed data for any commercial use.
