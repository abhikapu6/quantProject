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
- Confirm the number of cointegrating relationships (expect: 1)

### Deliverable
- Markdown summary: do both tests confirm cointegration? At what confidence level?
- Extract and store the **static hedge ratio** `β` from Engle-Granger OLS for use in Step 6

---

## Step 6: Static Spread Construction & Analysis

### Spread Construction
- `static_spread = log_GLD - β * log_GDX - α`
- Plot spread over time

### Z-Score
- `z_score = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()`
- Use a 60-day rolling window initially (revisit after estimating half-life)
- Plot z-score with ±2 and ±0.5 threshold lines overlaid

### ADF on Spread
- Run ADF on the static spread — confirm stationarity
- Report stat and p-value in a markdown cell

### Half-Life Estimation
- Fit AR(1) on spread: `Δspread_t = λ·spread_{t-1} + ε`
- `half_life = -ln(2) / ln(1 + λ)`
- Report half-life in trading days — this is your natural mean-reversion window
- Use half-life as the rolling window for z-score in Step 8 (Kalman)

### ACF/PACF
- Plot `plot_acf` and `plot_pacf` for the static spread (lags=60)
- Caption: confirm autocorrelation structure consistent with mean-reversion

---

## Step 7: Kalman Filter — Dynamic Hedge Ratio

This is the analytical differentiator. The static OLS hedge ratio assumes the relationship between GLD and GDX is fixed — the Kalman filter relaxes this.

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

### Plots
- **Hedge ratio over time:** static (flat line) vs. dynamic (Kalman estimate) — key visualization
- **Spread comparison:** static spread vs. dynamic spread on the same chart
- Caption: the dynamic spread should track more tightly and show less drift, especially around structural breaks (2020, 2022)

---

## Step 8: Signal Generation

### Z-Score on Dynamic Spread
- Compute rolling z-score using `half_life` (from Step 6) as the window
- Thresholds:
  - `z > +2.0` → **short spread** (GLD rich relative to GDX)
  - `z < -2.0` → **long spread** (GLD cheap relative to GDX)
  - `|z| < 0.5` → **close position**

### Signal Column
- Create a `signal` column: +1 (long), -1 (short), 0 (flat)
- Track position changes to identify entry/exit dates

### Plots
- Z-score time series with ±2.0 and ±0.5 horizontal lines
- Entry/exit markers overlaid on the normalized price chart
- Caption: note that we are not backtesting PnL — signals are evaluated for economic coherence, not profitability

---

## Step 9: Regime Analysis

This is the "depth of thinking" section — explain *why* the relationship behaves differently at different times.

### Regimes to Highlight
- **2011–2012:** Divergence — gold peaks while miners underperform (cost inflation, operational issues in mining sector)
- **2020 (COVID crash):** Temporary breakdown — both sell off sharply but at different rates; cointegration stress
- **2022 (rate hike cycle):** Both assets under pressure from rising real rates; spread behavior changes
- **2023–2024:** Re-anchoring as gold makes new highs

### Rolling Cointegration Strength
- Compute rolling ADF p-value on the spread using a 252-day (1-year) rolling window
- Plot over time — periods where p-value rises above 0.05 indicate cointegration breakdown
- Overlay as shaded regions on the price/spread chart

### Commentary
- For each regime: what was the macro driver? did it affect GLD and GDX symmetrically? what happened to the spread?
- This section should be prose-heavy with figures inline — it's graded on depth of interpretation

---

## Step 10: Model Evaluation & Comparison

### Static vs. Kalman Spread Comparison Table

| Metric | Static Spread | Dynamic Spread |
|---|---|---|
| ADF Statistic | ... | ... |
| ADF p-value | ... | ... |
| Half-Life (days) | ... | ... |
| Rolling Std (full) | ... | ... |
| Spread Range | ... | ... |

- The Kalman spread should show a lower ADF p-value (more stationary), shorter half-life, and tighter range
- If you add a simple ARIMA forecast on the spread, report RMSE on a holdout period (optional but adds points)

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
3. **Key Findings** — 3–4 bullet points: are they cointegrated? what is the half-life? how does the dynamic hedge ratio improve the spread? what regimes broke down?
4. **Figures** — include at least: normalized prices, dynamic hedge ratio over time, z-score with signals
5. **Reflection** — what would you do next? what are the limitations? (static threshold, no transaction costs, look-ahead bias in Kalman init)

---

## Step 13: Presentation Slide

**Best visualization:** Dynamic z-score with signal thresholds overlaid  
**3 bullet takeaways:**
- GLD and GDX are cointegrated over 2010–2024, with a mean-reversion half-life of ~X days
- A Kalman filter hedge ratio outperforms a static OLS ratio — tighter, more stationary spread
- Cointegration breaks down around macro stress events (2020, 2022) and recovers — regime awareness is critical for live deployment

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
