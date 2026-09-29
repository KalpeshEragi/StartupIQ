# StartupIQ — AI Startup Evaluator

## Identity
StartupIQ is an AI-powered startup intelligence and decision-support platform. It should feel like an intelligent startup analyst, not a form + classifier.

## Purpose & Problem
Help users understand a startup's potential, readiness, risks, strengths, weaknesses, market/business position, investment suitability and incubation suitability, even when information is sparse.

## Stakeholders
1. Founders / entrepreneurs
2. Investors / angels / VCs
3. Incubators / accelerators
(Details: docs/03-stakeholders-and-use-cases.md)

## Core User Journey
User → startup information → startup profile → missing-info detection → targeted research → AI/ML analysis → risk/strength analysis → decision engine → explainable recommendation → investor matching → incubator matching → report/dashboard → optional continuous monitoring.
- Must work from basic info alone; never assume a website, pitch deck or complete data.
- Inputs: conversational text, structured questions, website, pitch deck/document, optional voice.
- Ask only useful follow-ups; allow "unknown"; research only when appropriate.
- NEVER invent unknown startup information.

## Main Capabilities
Startup evaluation, explainable assessment, risk analysis, investor matching, incubator matching, evidence-backed insights, actionable recommendations, optional monitoring.

## Intelligence Approach
Right tool per job, no buzzwords:
- ML: prediction from historical data
- LLM/NLP: text, documents, descriptions, research
- Rules: deterministic constraints and business logic
- Recommendation: investor/incubator matching
- RAG: evidence-backed retrieval where useful
- Agents: only where they add genuine value

## Output
Multidimensional assessment, not a single label: overall assessment, outcome prediction, investment readiness, growth potential, risk, strengths, weaknesses, evidence, recommendation. Action outcomes: INVEST / INCUBATE / REJECT. These are NOT assumed to be the raw ML target; the decision may combine ML, rules, profile, stakeholder requirements and evidence.

## Data Philosophy
- No single dataset powers everything; different sources serve different modules (company history, funding, investors, incubators, market, documents/websites, live external intelligence).
- Integrate logically; never blindly merge incompatible datasets.
- Synthetic data only for testing/demo, never presented as real-world evidence.

## Explainability (core requirement)
Every important recommendation states: conclusion, why, positive factors, negative factors, supporting evidence, and separates facts / predictions / rules / external research.

## Automation & Agents
- Agent count not fixed; choose the simplest architecture with real value (candidates: Startup Analyst, Market Research, Financial/Risk, Matching, Orchestrator).
- Market research is targeted and cost-conscious: reuse knowledge, research only gaps, prefer reliable sources, store evidence, don't repeat research.
- Automation only for real needs: periodic monitoring, market/competitor/funding/news changes, re-evaluation, notifications.

## Final Vision
A deployable web application delivering the flow above.

## Constraints
- Requirements are stable; implementation choices stay open until deliberately designed.
- Do not lock stack, DB, models, agents, APIs, frontend, deployment prematurely.
- Academic practicals are building blocks of this system, not mini-projects (docs/08).
- No web research or invented requirements; no implementation code or detailed architecture yet.

## Open Decisions
See docs/09-open-decisions.md (architecture, stack, data sources, models, agent design, storage, deployment, monitoring scope).

## Docs Index
docs/01-product-vision · 02-user-journey · 03-stakeholders-and-use-cases · 04-intelligence-and-ai · 05-data-strategy · 06-explainability-and-decision-engine · 07-automation-and-agents · 08-academic-project-alignment · 09-open-decisions
