# StartupIQ design system

The source of truth for every screen. Read before building or changing UI; hold to these values.

## Intent

- **Who:** an Indian founder (seed to Series A) between investor calls, opening the app once a week and properly
  once a month after downloading the bank statement. Not a finance person; numerate, short on time.
- **Task:** know where the company stands this month (revenue, burn, runway), fix what is unsorted, answer an
  investor's question with a number they trust.
- **Feel:** *a well-edited monthly brief on good paper.* Calm and confident like an investor letter (Editorial
  Ledger), friendly and modular like a modern product (Indigo Bento). Never a trading terminal, never a toy.

## Domain

- Vocabulary: ledger, statement, the month-end close, runway, burn, cap table, round, term sheet, brief, memo.
- Color world: paper and ink of a printed annual report; the indigo of a fountain-pen signature on a term sheet;
  the green of a cleared cheque / profit column; the amber of a "for review" sticky flag; rupee notes' muted tones.
- **Signature: the monthly brief.** Every workspace opens with a sentence written from the startup's own ledger
  ("Chai Labs earned ₹17 L in November and kept more cash than it spent.") set in the serif, with an amber kicker
  line above it. Figures are typeset like a newspaper's, not like a widget.
- Rejected defaults: generic KPI boxes in a row → one hero tile with the story plus smaller supporting tiles;
  Inter everywhere → Newsreader serif for headlines and figures, Work Sans for interface; colored sidebar →
  sidebar on the same paper as the canvas, separated by a rule.

## Tokens (CSS variables, `frontend/src/index.css`)

| Role | Light | Dark | Why |
| --- | --- | --- | --- |
| `--bg` paper (canvas) | `#f7f6fd` | `#0b0d16` | lavender-white paper; one hue family with the indigo |
| `--panel` sheet (cards) | `#ffffff` | `#131624` | a clean sheet laid on the paper |
| `--panel-2` inset / table head | `#faf9ff` | `#171b2b` | |
| `--ink` primary text | `#1e1b4b` | `#eef0f7` | indigo-black ink, never pure black |
| `--ink-2` secondary | `#565a73` | `#a3a9bb` | |
| `--ink-3` meta / muted | `#6b6e88` | `#8a90a8` | 4.6:1+ on paper and sheet (AA) |
| `--line` rule | `#e6e4f5` | `#232839` | hairline ledger rules |
| `--line-2` soft rule | `#f0eefb` | `#1c2031` | |
| `--accent` indigo (signature ink, actions) | `#5458e8` | `#8b80ff` | the one accent; white text on it 5.3:1; hover `#4f46e5` |
| `--good` emerald (profit, positive) | `#047857` | `#34d399` | meaning only, never decoration; 5.1:1 |
| `--warn` amber (needs review, kicker) | `#b45309` | `#fbbf24` | the "for review" flag and the brief's kicker; 4.7:1 |
| `--bad` rose (loss, error) | `#c81e3a` | `#fb7185` | 5.3:1 |

Every text colour passes WCAG AA (4.5:1) on paper and sheet; no text below 12px.
No gradients anywhere in the app (flat fills only).

Rule: ~60% paper/sheet, ~30% ink and rules, ~10% indigo. Emerald/amber/rose only when they carry meaning.

## Typography

- **Newsreader** (serif, 500-700): page titles, the brief headline and lede, card titles (h2), every headline figure
  (KPI values, metric values, table totals in hero contexts). Tracking -0.02em at 28px+.
- **Work Sans** (400-700): everything interactive and every label.
- Scale (ratio ~1.25, base 15): `12 meta · 13 label · 15 body · 18 h3 · 20 h2 (serif) · 30 figure (serif) ·
  34 page title (serif) · 42 brief headline (serif) · 58 hero figure (serif)`.
- Three tiers at one size by weight + colour: value 600/ink, label 600/ink-2 at 13px, meta 400/ink-3 at 12px.
- Numbers: `font-variant-numeric: tabular-nums` everywhere they can change. Rupees via `en-IN` grouping
  (₹16,94,264) in tables, compact (₹17 L, ₹2.0 Cr) in tiles.
- Kicker: 12px / 700 / 0.14em tracking / uppercase / amber, items separated by small dots.

## Spacing and shape

- Base unit 4px. Component padding 12-16, card padding 22, tile gap 16, section gap 24-28, page gutter 32 (16 on
  mobile). Content max width 1240px. Sidebar 260px.
- Radius scale: 8 small controls · 10 buttons/inputs · 20 cards · 24 bento tiles · 99 pills/avatars.
- Depth: **subtle shadows** (approachable), one strategy: `0 1px 2px rgba(15,18,34,.04), 0 4px 16px rgba(15,18,34,.04)`;
  hover lift `0 12px 40px rgba(15,18,34,.10)`. Dark mode: no depth shadow, rely on the rule colour.
- Rules over boxes inside cards (dashed rule above inline forms; hairline between list rows).

## Layout

- Shell: fixed left sidebar (brand, workspace switcher, sections, "coming next", user + sign out) on the same paper;
  sticky translucent top bar with breadcrumb, currency badge, theme toggle, avatar. Below 1000px the sidebar
  becomes a drawer with a scrim.
- Overview: brief (kicker, serif headline, serif lede, actions) → bento grid (4 columns): hero tile 2x2 in indigo
  (revenue + 6-month bars), supporting tiles (EBITDA, margin, burn, review flag), chart 3 columns, "where the money
  went" 1 column → unit economics → company + activity.
- Other pages: page head (serif title + one-line muted explanation, actions right) → cards.

## Components

- **Button** primary: 38px h · 9px 16px · radius 10 · Work Sans 14/600 · indigo fill + soft indigo shadow; hover
  lighter indigo; active translateY(1px). Secondary: sheet + rule border. Ghost: transparent, ink-2. Small: 13px,
  6px 11px, radius 8.
- **Icon button:** 34px square, radius 9, ink-2 → ink on hover with soft-rule fill. Always `aria-label`.
- **Input/select:** 15px text · 9px 12px · radius 10 · rule border · focus = indigo border + 4px indigo-soft ring.
  Label above (13/600), hint below (12, ink-3), error box near the form (rose on rose-soft).
- **Card:** sheet, rule border, radius 20, padding 22, subtle shadow. Title = serif h2 20px.
- **Bento tile:** card at radius 24; stat-top row = label (13/600 ink-2) + 36px tinted icon chip (radius 11);
  value in serif 30px; delta pill (emerald/rose soft); note 12px ink-3. Hero tile: indigo, white serif 58px figure.
- **Review flag tile/banner:** amber-soft fill, amber figure, one action.
- **Table (`.pnl`):** uppercase 11.5px tracked header on inset band with rounded ends; 10px 14px cells; hairline row
  rules; row hover inset; section rows in indigo caps; totals bold.
- **Chips:** indigo-soft pill with indigo text, fills indigo on hover. **Pills/badges:** 12px/600 on soft fill.
- **Kanban column:** inset panel, radius 14; deal cards are sheets with the subtle shadow.

## States (required on every screen)

- Interactive: default, hover, active, focus-visible (indigo ring), disabled (55% opacity).
- Data: loading (muted text "Loading…" or skeleton), empty (icon chip + serif title + one sentence + primary
  action), error (rose box near the content, with a way back).
- Motion: 150-200ms on colour/shadow/transform only, ease-out; respect `prefers-reduced-motion`.
