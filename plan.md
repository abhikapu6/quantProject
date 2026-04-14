# Quant Final Project Plan: Pairs Trading — GLD/GDX Cointegration Analysis

## Dataset
- **Tickers:** GLD (SPDR Gold Trust ETF) and GDX (VanEck Gold Miners ETF)
- **Source:** Yahoo Finance via `yfinance`
- **Period:** 2010-01-01 to 2024-12-31 (daily OHLCV)
- **Rationale:** GLD tracks spot gold; GDX tracks gold mining companies. Both are driven by gold prices but GDX carries additional equity/operational risk, making the spread economically meaningful and non-trivial.

---

## Step 1: Environment Setup & Data Collection

**Libraries:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `pykalman`, `scipy`

### Tasks
- Install and import all dependencies at the top of the notebook
- Pull daily OHLCV for GLD and GDX via `yfinance.download()` for 2010–2024
- Keep only **Adjusted Close** prices (accounts for splits and dividends)
- Print `.shape`, `.dtypes`, `.head()`, `.tail()` to document the raw data
- Save raw data to CSV as a reproducibility checkpoint

### Cleaning Checklist
- Check for missing values: `df.isnull().sum()` — print a missing value report
- Align both series to the **same trading calendar** — inner join on date index to drop days where either ticker has no data
- Forward-fill any remaining isolated NaNs (e.g., single-day data gaps); document this choice explicitly in a markdown cell
- Drop any rows still containing NaN after forward-fill and log the count
- Assert no NaNs remain before proceeding

### Deliverable
- Clean `pd.DataFrame` with columns `['GLD_adj', 'GDX_adj']`
- Markdown cell documenting: date range, final shape, missing value count before/after cleaning, and fill method used

---

## Step 2: Log-Price Transformation

- Compute `log_GLD = np.log(GLD_adj)` and `log_GDX = np.log(GDX_adj)`
- Add both as columns to the main DataFrame
- Brief markdown explanation of *why* log prices: variance stabilization, log-returns are additive, standard in financial econometrics
- Compute **log returns** as well: `returns = log_prices.diff().dropna()` — needed for ADF on first differences in Step 3

---

## Step 3: EDA — Visualizations

Produce the following plots. Each must have a title, labeled axes, and a markdown caption explaining what to look for.

### Plot 1: Normalized Price Series
- Divide each series by its first value so both start at 1.0
- Single chart, both tickers, different colors, legend
- Caption: highlight divergence periods and general co-movement

### Plot 2: Rolling 60-Day Correlation
- Compute `returns.rolling(60).corr()` between GLD and GDX returns
- Plot over time with a horizontal reference line at the full-sample mean correlation
- Caption: note periods where correlation drops (potential regime shifts)

### Plot 3: Rolling 30-Day Volatility
- Compute rolling 30-day std of returns for each ticker
- Plot both on the same chart
- Caption: compare volatility spikes — GDX should be more volatile (equity risk layered on gold risk)

### Plot 4: Log Price Ratio (Naive Spread)
- Compute `log_GLD - log_GDX` and plot over time
- This is a first visual check for mean-reversion before any formal modeling
- Caption: does it look stationary? does it drift? this motivates the formal tests in Step 4

### Plot 5: Return Distribution
- Side-by-side histograms (or KDE) of daily log returns for GLD and GDX
- Overlay a normal distribution fit
- Caption: note fat tails — relevant when interpreting signal thresholds later

---

## Step 4: Stationarity Testing (ADF)

Goal: establish both log price series are I(1) — integrated of order 1 — which is required for cointegration to be meaningful.

### Tests to Run
For each of `log_GLD`, `log_GDX`, `delta_log_GLD` (first diff), `delta_log_GDX`:
- Run `statsmodels.tsa.stattools.adfuller()` with `autolag='AIC'`
- Extract: test statistic, p-value, number of lags used, critical values (1%, 5%, 10%)

### Deliverable
A clean markdown table:

| Series | ADF Stat | p-value | Lags | 5% Critical | Conclusion |
|---|---|---|---|---|---|
| log_GLD | ... | ... | ... | ... | Non-stationary |
| log_GDX | ... | ... | ... | ... | Non-stationary |
| Δlog_GLD | ... | ... | ... | ... | Stationary |
| Δlog_GDX | ... | ... | ... | ... | Stationary |

- Interpretation: both log price series are I(1); first differences (returns) are I(0). This satisfies the prerequisite for cointegration testing.

---

## Step 5: Cointegration Testing

### Engle-Granger Test
- OLS regress `log_GLD` on `log_GDX` (and a constant): `log_GLD = α + β·log_GDX + ε`
- Extract: intercept `α`, hedge ratio `β`, R²
- Run ADF on the residuals `ε` — if stationary, the pair is cointegrated
- Report ADF stat and p-value on residuals

### Johansen Test
- Use `statsmodels.tsa.vector_ar.vecm.coint_johansen(df[['log_GLD','log_GDX']], det_order=0, k_ar_diff=1)`
- Report trace statistic and max-eigenvalue statistic vs. critical values
- Confirm the number of cointegrating relationships

### Actual Result & Interpretation
**Both tests find NO cointegration over the full 2010–2024 period** (EG p=0.45, Johansen trace 10.59 vs 15.49 critical). This is an honest and economically meaningful result: GLD approximately doubled while GDX declined ~40% — a permanent structural divergence driven by the de-rating of gold mining equities. The log price ratio (Plot 4) already signalled this via its persistent upward trend.

### Step 5b: Rolling Cointegration Analysis (added in response to full-period result)
- Compute rolling Engle-Granger p-value on a **504-day (2-year) sliding window**
- Also compute rolling OLS β over the same window to track hedge ratio instability
- Plot both panels: p-value over time with 5% threshold shaded, and rolling β vs. static β
- **Result:** cointegrated in 79.3% of rolling windows; β ranged from -0.10 to 1.15 — confirming the relationship existed historically but is structurally unstable
- This directly motivates the Kalman filter: a single static β cannot capture a parameter that drifts by over 1.2 units across the sample

### Deliverable
- Full-period test results reported honestly with economic explanation of why non-cointegration holds
- Rolling cointegration chart showing when the relationship was and wasn't active
- Store OLS `β` and `α` as `STATIC_BETA` / `STATIC_ALPHA` — used as baseline in Step 6

---

## Step 6: Static Spread Construction & Analysis

**Purpose:** Build the static OLS spread as an explicit *baseline* that we expect to be imperfect. Its non-stationarity over the full period is the problem that the Kalman filter solves. Present it honestly rather than pretending it works.

### Spread Construction
- `static_spread = log_GLD - STATIC_BETA * log_GDX - STATIC_ALPHA`
- Plot spread over time — expect a visible upward trend confirming non-stationarity

### ADF on Static Spread
- Run ADF on the full-period static spread — **expect non-stationary** (p > 0.05) given the full-period non-cointegration result
- Report stat and p-value; frame this as confirming what Step 5 already showed
- This result directly motivates moving to the dynamic approach in Step 7

### Half-Life Estimation (interpret with caution)
- Fit AR(1) on spread: `Δspread_t = λ·spread_{t-1} + ε`
- `half_life = -ln(2) / ln(1 + λ)`
- If the spread is non-stationary, λ will be close to 0 and half-life will be very long or undefined — **report this and explain what it means**
- We will use the Kalman spread's half-life (Step 7) as the operative window for signal generation

### Z-Score (for visual purposes only on static spread)
- Plot rolling z-score with ±2 and ±0.5 thresholds
- Note explicitly: because the underlying spread trends, z-score thresholds fire asymmetrically — another symptom of the non-stationarity problem

### ACF/PACF
- Plot `plot_acf` and `plot_pacf` for the static spread (lags=60)
- Caption: if autocorrelation decays slowly, this is additional evidence of non-stationarity

---

## Step 7: Kalman Filter — Dynamic Hedge Ratio

**This is now the central analytical contribution.** Step 5 proved that a static hedge ratio is structurally inadequate: β drifted from -0.10 to 1.15 across the sample. The Kalman filter is the correct solution — it estimates a time-varying β that continuously adapts to structural shifts in the GLD/GDX relationship.

### Narrative framing
The Kalman filter doesn't just "improve" the spread — it solves a specific identified problem. Make this connection explicit in the markdown: "Step 5 showed the static β is unstable; Step 7 addresses this directly."

### State Space Formulation
- **Observation equation:** `log_GLD_t = β_t · log_GDX_t + α_t + ε_t`
- **State transition:** `[β_t, α_t] = [β_{t-1}, α_{t-1}] + η_t` (random walk prior on parameters)
- Hidden state: `[β_t, α_t]` — time-varying hedge ratio and intercept

### Implementation with `pykalman`
- Use `KalmanFilter` from `pykalman`
- Initialize transition and observation covariance matrices
- Fit using `.em()` (expectation-maximization) to learn covariance parameters from data
- Extract filtered state means: `hedge_ratio_dynamic`, `intercept_dynamic`

### Dynamic Spread
- `dynamic_spread = log_GLD - hedge_ratio_dynamic * log_GDX - intercept_dynamic`
- Run ADF on the dynamic spread — **expect more stationary than static spread**
- Estimate half-life of the dynamic spread — use this as the operative window for Step 8

### Plots
- **Hedge ratio over time:** static (flat line) vs. dynamic (Kalman estimate) — key visualization showing the structural drift the static model missed
- **Spread comparison:** static spread vs. dynamic spread — dynamic should show less trend and tighter range
- Caption: connect the hedge ratio chart to the rolling β from Step 5b — the Kalman estimate should track the rolling OLS β but with smoother, lag-free adaptation

---

## Step 8: Signal Generation

**Signals are only generated during confirmed cointegration windows.** Generating signals on a non-cointegrated spread is economically incoherent — there is no mean-reversion to exploit. Use the rolling p-value from Step 5b to gate signal output.

### Z-Score on Dynamic Spread
- Compute rolling z-score using the **Kalman spread's half-life** (from Step 7) as the window
- Thresholds:
  - `z > +2.0` → **short spread** (GLD rich relative to GDX)
  - `z < -2.0` → **long spread** (GLD cheap relative to GDX)
  - `|z| < 0.5` → **close position**

### Regime-Aware Signal Gating
- Mask signals: set signal to 0 in any window where the rolling cointegration p-value ≥ 0.05
- This means signals only fire when there is statistical evidence that mean-reversion is active
- Plot the gated signal count by year — expect fewer signals in 2022–2024 when cointegration weakened

### Signal Column
- Create a `signal` column: +1 (long), -1 (short), 0 (flat/gated)
- Track position changes to identify entry/exit dates

### Plots
- Z-score time series with ±2.0 and ±0.5 horizontal lines, with non-cointegrated periods shaded grey
- Entry/exit markers overlaid on the normalized price chart (only in active cointegration windows)
- Caption: signals are evaluated for economic coherence, not PnL — the regime gating is the key addition vs. a naive threshold approach

---

## Step 9: Regime Analysis

**This is now the analytical centerpiece of the project.** The rolling cointegration chart from Step 5b already does the quantitative work — this section adds the economic narrative that explains *why* the relationship behaved the way it did. Graded on depth of interpretation.

### The Central Story
The full-period non-cointegration result is not a failure — it reveals that the GLD/GDX relationship is **regime-dependent**. The relationship was active and tradeable for extended periods, but structural macro forces periodically broke it down. Identifying those regimes is the interesting analytical question.

### Regimes to Explain (connect to rolling p-value chart from Step 5b)
- **2010–2012 (Active):** Gold supercycle peak — GLD and GDX moved tightly together as gold prices drove miner revenues directly. Cointegration strong. β high (miners had operating leverage to gold price).
- **2013–2015 (Breakdown begins):** Gold crash post-2011 peak. Miners hit harder than physical gold due to high fixed costs, write-downs, and capex overhang — GDX fell ~70% from peak while GLD fell ~40%. β begins structural decline.
- **2016–2019 (Partial recovery):** Gold stabilises; some windows re-cointegrate but relationship is weaker. Rolling β in a lower range (~0.3–0.6).
- **2020 (COVID stress):** Both assets shocked simultaneously — initial correlation spike followed by divergence (gold rallies as safe haven, miners lag due to operational shutdowns). Brief cointegration breakdown.
- **2022 (Rate hike cycle):** Rising real rates hit both assets but differently. GDX also absorbs equity beta and energy cost inflation. Cointegration weakest in this window.
- **2023–2024:** Gold makes new all-time highs on central bank buying. GDX partially recovers but lags — some windows re-cointegrate.

### Plots for this section
- **Reference** the rolling p-value + rolling β chart from Step 5b — do not replot
- Add: normalized prices with shaded bands where rolling p-value ≥ 0.05 (non-cointegrated periods)
- Add: summary table — period, cointegrated Y/N, approximate β range, macro driver

### Commentary requirement
Each regime entry must answer: (1) what was the macro driver? (2) did it affect GLD and GDX symmetrically or asymmetrically? (3) what did that mean for the spread? Write 3–5 sentences per regime.

---

## Step 10: Model Evaluation & Comparison

### Static vs. Kalman Spread Comparison Table

| Metric | Static Spread | Dynamic Spread |
|---|---|---|
| ADF Statistic | ... | ... |
| ADF p-value | ... | ... |
| Half-Life (days) | ... (likely very long / undefined) | ... (expect shorter) |
| Rolling Std (full) | ... | ... |
| Spread Range | ... | ... |

- The static spread is expected to be non-stationary (p > 0.05) — this is the baseline that confirms the problem
- The Kalman spread should be more stationary (lower p-value), have a shorter half-life, and a tighter, less-drifting range
- Frame the comparison as: "static OLS fails because it cannot track the structural shift; Kalman succeeds because it was designed for exactly this"
- If you add a simple ARIMA forecast on the Kalman spread, report RMSE on a holdout period (optional but adds points)

---

## Step 11: Notebook Polish

Before finalizing:
- Every section starts with a `## Section Title` markdown header
- Every section has 2–4 sentences of markdown *before* any code explaining what and why
- All functions are defined with docstrings; no repeated blocks of copy-pasted code
- All plots: title, x-label, y-label, legend where needed, `plt.tight_layout()`
- Variable names are descriptive (`log_gld_prices` not `x`)
- Remove all dead/commented-out code cells
- Restart kernel and run all cells top-to-bottom to confirm full reproducibility
- First cell: pip install block with version pins for key libraries

---

## Step 12: PDF Report Outline

**Length:** 2–3 pages  
**Audience:** smart non-technical reader (investment committee member)

1. **Dataset & Motivation** — what is GLD/GDX, why is this pair interesting, what period
2. **Methods Overview** — EDA, stationarity testing, cointegration (brief), Kalman filter (intuition only, no math)
3. **Key Findings** — lead with the non-cointegration result as the interesting discovery, not a caveat:
   - Full-period cointegration does not hold — GDX structurally de-rated relative to GLD
   - Rolling analysis shows cointegration was active ~79% of 2-year windows — the relationship is real but regime-dependent
   - Static hedge ratio is unstable (β range -0.10 to 1.15) — motivates dynamic estimation
   - Kalman filter produces a more stationary spread with a shorter half-life
   - Regime-aware signal gating produces more economically coherent entry/exit points
4. **Figures** — include at least: normalized prices with regime shading, dynamic hedge ratio vs. static, z-score with gated signals
5. **Reflection** — limitations: look-ahead bias in Kalman initialisation, no transaction costs, threshold not optimised, non-cointegration in recent years means the strategy may not be currently deployable

---

## Step 13: Presentation Slide

**Best visualization:** Normalized prices with regime shading overlaid (non-cointegrated periods highlighted), or the dynamic hedge ratio showing the structural drift  
**3 bullet takeaways:**
- GLD/GDX cointegration is regime-dependent: active in ~79% of rolling 2-year windows but absent over the full 2010–2024 period — a structural de-rating of gold mining equities is the driver
- A static hedge ratio is inadequate: β drifted from −0.10 to 1.15 across the sample; a Kalman filter tracks this shift and produces a more stationary spread
- Regime-aware signal gating (signals only when cointegration is confirmed) is economically more coherent than naive z-score thresholds applied unconditionally

---

## File Structure

```
quantProject/
├── plan.md                  # this file
├── data.py                  # data fetching utilities (optional)
├── notebook.ipynb           # main deliverable
├── data/
│   ├── gld_gdx_raw.csv      # raw download checkpoint
│   └── gld_gdx_clean.csv    # cleaned data checkpoint
└── report/
    ├── report.pdf
    └── slide.pdf
```

---

## Rubric Mapping

| Rubric Category | Steps | Target Score |
|---|---|---|
| Data Collection & Cleaning (20 pts) | 1, 2 | 20/20 |
| EDA & Visualizations (25 pts) | 3, 6 | 25/25 |
| Modeling / Statistical Analysis (20 pts) | 4, 5, 7, 10 | 20/20 |
| Insights & Interpretation (20 pts) | 8, 9 | 20/20 |
| Code Quality (15 pts) | 11 | 15/15 |
