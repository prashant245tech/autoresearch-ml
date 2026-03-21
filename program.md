# program.md — ML Experiment Configuration
# ═══════════════════════════════════════════════════════════════
# This is the ONLY file you edit as a human.
#
# HOW IT WORKS:
#   1. Fill in SECTION 1 — data file path and target column
#   2. Describe your features in plain English in SECTION 2
#   3. Run: python compile.py
#      → reads your descriptions, calls Claude, produces feature_spec.json
#   4. Run: python prepare.py  → cleans and splits your data
#   5. Run: python train.py    → baseline experiment
#   6. Iterate manually or: python run_experiment.py --n 10 --auto
#
# SECTION 3 gives the AI agent domain context for smarter experiments.
# ═══════════════════════════════════════════════════════════════


## ── SECTION 1: REQUIRED CONFIG ─────────────────────────────────────────────

```config
data_file: /Users/prashant/Downloads/Tier_TreeHouse_Winland 1 - Copy.xlsx
target_column: Tier 1 Price/MSF
test_size: 0.20
random_state: 42
outlier_iqr_k: 4.0
```


## ── SECTION 2: FEATURE DESCRIPTIONS (plain English) ───────────────────────
##
## Describe each feature you want to use in natural language.
## compile.py reads these and produces exact cleaning rules in feature_spec.json.
## Be as descriptive as you like — mention value ranges, missing value strategy,
## whether it's a category/ranking/flag, and any domain relationships.

```features

Size Bucket:
  Pre-bucketed size band for the box based on dimensional scale.
  Ordered from smallest to largest: 0-3.0, 3.1-5.5, 5.6-8.5, 8.6-15.0, 15.1-30.0, 30.1-90.0.
  Treat unexpected or null values as unknown sentinel -1.

SQ. FT. PER PC:
  Surface area of the corrugated blank in square feet per piece.
  The single most important pricing driver — more area means more board consumed.
  Fill missing with column median. Clip to minimum 0.01.

Quantity:
  Order quantity in pieces. Many rows have zero — these are catalog or model prices,
  not actual orders. Keep zero rows but the distinction matters for the model.
  Fill missing with 0. Clip negative values to 0.

Flute 1:
  Flute type — an ordered ranking by board thickness and cost.
  Order from thinnest/cheapest to thickest/most expensive: F, E, N, B, C, BC.
  BC is double-wall and costs roughly 40-60% more per MSF than B-flute.
  Normalise inconsistent casing and spaces. Unknown values get a sentinel of -1.

Stock:
  Liner type. Either KRAFT (natural brown, cheaper) or WHITE (bleached, more expensive).
  Normalise messy casing variants (wHITE, white → WHITE). Anything not clearly
  KRAFT or WHITE should be treated as OTHER. Fill missing with UNKNOWN.

Box Style:
  Structural style of the box. Main families: RSC, DCT, DCJ, TRAY, HSC, SHT, PAD.
  Consolidate messy variants: D/C JOINED → DCJ, D/C NON JOINED → DCJ,
  Die Cut → DCT, D/C RSC → RSC, D/C HSC → HSC. Unknown → OTHER.
  Fill missing with UNKNOWN.

Region:
  Geographic sales region — Midwest, Northeast, Southeast, Southwest, West, Canada.
  Affects price due to freight costs and regional supplier pricing differences.
  Normalise case. Fill missing with UNKNOWN.

Ink Coverage Bucket:
  Pre-bucketed print coverage level ordered from least to most ink:
  0-10%, 10-20%, 20-30%, 30-40%, 40-50%, >50%.
  Treat Null or unexpected values as unknown sentinel -1.

Tare Weight:
  Weight of the empty box in pounds. Correlates with board weight and caliper.
  Fill missing with column median. Clip to minimum 0.001.

Adhesive:
  Adhesive type used in box manufacture. MRA is standard, REGULAR is basic,
  WPA is a specialty type. Normalise casing. Fill missing with UNKNOWN.

Litho Box:
  Binary flag — 1 if this is a lithographic (premium printed) box, 0 otherwise.
  Litho boxes command a significant price premium. Fill missing with 0.

```


## ── SECTION 3: DOMAIN CONTEXT FOR THE AI AGENT ────────────────────────────

### What we are predicting
**Tier 1 Price/MSF** = price per thousand square feet of corrugated board
for the primary tiered quote level we want to model first.
Industry-standard pricing unit for corrugated packaging in North America.

### Dataset
- Source: TreeHouse Foods / Winland supplier pricing data
- ~2,950 usable rows after cleaning
- Mix of RSC, DCT, DCJ, TRAY, HSC box styles
- Predominantly B-flute and C-flute; small number of BC double-wall

### Key pricing drivers (in rough order of importance)
1. **Board area (SQ. FT. PER PC)** — primary cost driver; larger boxes = lower $/MSF
   due to setup cost amortisation over more board.
2. **Quantity** — strong inverse log relationship. Price breaks at 500, 1k, 2.5k, 5k, 10k+.
   Zero-qty rows behave differently (catalog prices) — flag them.
3. **Size Bucket** — compact proxy for carton scale when using bucketed sizing instead of raw dimensions.
4. **Flute type** — cost hierarchy: F < E < N < B < C < BC.
5. **Printing** — Ink Coverage Bucket and Litho Box add meaningful cost.
6. **Stock** — WHITE liner costs more than KRAFT.
7. **Box Style** — die-cut styles have higher setup cost than RSC.
8. **Region** — freight and regional supplier pricing variation.

### Derived features worth trying in train.py
```python
# High value
log_sqft        = np.log1p(df["SQ. FT. PER PC"])
qty_log         = np.log1p(df["Quantity"])
has_quantity    = (df["Quantity"] > 0).astype(int)
area_x_qty      = df["SQ. FT. PER PC"] * qty_log
area_x_flute    = df["SQ. FT. PER PC"] * df["Flute 1_encoded"]
area_x_size     = df["SQ. FT. PER PC"] * df["Size Bucket_encoded"]
area_x_ink      = df["SQ. FT. PER PC"] * df["Ink Coverage Bucket_encoded"]

# Medium value
weight_per_sqft = df["Tare Weight"] / (df["SQ. FT. PER PC"] + 1e-6)
setup_proxy     = 1.0 / (df["SQ. FT. PER PC"] * qty_log + 1.0)
qty_tier        = pd.cut(df["Quantity"],
                    bins=[-1,0,500,1000,2500,5000,10000,np.inf], labels=False)
```

### Agent decision rules
1. Read experiments/results.json before each run — know the current best MAPE.
2. Hypothesis first — state what you expect and why before modifying train.py.
3. One change at a time — features OR model, not both simultaneously.
4. Keep if MAPE improves; revert if it worsens.
5. After 5 failed feature experiments, shift focus to model tuning.

### Target thresholds
| MAPE    | Assessment          |
|---------|---------------------|
| > 20%   | Poor                |
| 15–20%  | Needs improvement   |
| 10–15%  | Acceptable          |
| 5–10%   | Good                |
| < 5%    | Excellent           |

### Stopping criteria
- Test MAPE < 8% for two consecutive experiments
- 25 total experiments with no improvement in last 8
- All feature ideas tried AND ≥ 3 model families tested
