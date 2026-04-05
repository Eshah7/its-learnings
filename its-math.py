import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # NWSL VAR & Foul Rates: Interrupted Time Series Analysis

    The NWSL introduced VAR in March 2023 — the first women's domestic league to do so.
    Did it actually change how the game is played?

    We use **three increasingly powerful methods** — OLS Segmented Regression, ARIMA with
    Intervention, and CausalImpact (BSTS) — and learn what each one adds.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Package Setup

    Install these once before running:

    ```bash
    pip install soccerdata pmdarima statsmodels pycausalimpact matplotlib pandas numpy
    ```

    `soccerdata` wraps FBref. `pmdarima` auto-selects ARIMA order.
    `pycausalimpact` is the Python CausalImpact implementation.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    return np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Pull Match-Level Data from ESPN Core API

    FBref blocks automated requests with 403. Instead we use **ESPN's Core API**
    — a JSON API that returns full match stats including fouls, cards, and
    penalties awarded. No scraping, no rate-limit issues.

    | League | ESPN slug | Seasons | Role |
    |--------|-----------|---------|------|
    | NWSL | `usa.nwsl` | 2021–2024 | **Treated** — VAR from March 2023 |
    | WSL | `eng.w.1` | 2021-22 – 2024-25 | **Control** — no VAR |

    **Two-step fetch per game:**
    1. Scoreboard (month-by-month) → game IDs + team IDs
    2. `competitions/{id}/competitors/{team_id}/statistics` → `foulsCommitted`,
       `yellowCards`, `redCards`, `penaltyKicksFaced` per team

    Rate: 0.3 s between requests. ~600 games × 2 teams = ~1 200 calls → ~6 min
    first run. Cached to `data/` so all subsequent runs are instant.
    """)
    return


@app.cell
def _(pd):
    import requests
    import time
    from pathlib import Path

    _H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    _RATE = 0.3

    def _get(url, params=None):
        time.sleep(_RATE)
        r = requests.get(url, headers=_H, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    # ── Step 1: collect all (game_id, date, home_team_id, away_team_id) ───
    def _get_game_index(espn_slug, year_months):
        """
        Fetch scoreboard month-by-month.
        year_months: list of "YYYYMM" strings covering the season.
        Returns list of dicts with game_id, date, season, team IDs.
        """
        base = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_slug}/scoreboard"
        games = []
        seen = set()
        for ym in year_months:
            data = _get(base, params={"dates": ym, "limit": 100})
            for ev in data.get("events", []):
                if ev["id"] in seen:
                    continue
                seen.add(ev["id"])
                comp = ev["competitions"][0]
                competitors = comp["competitors"]
                games.append({
                    "game_id": ev["id"],
                    "date": ev["date"][:10],
                    "season_year": ev["season"]["year"],
                    "team_ids": [c["id"] for c in competitors],
                })
        return games

    # ── Step 2: fetch per-team stats for one game ─────────────────────────
    _CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"

    def _get_team_stats(espn_slug, game_id, team_id):
        url = f"{_CORE}/{espn_slug}/events/{game_id}/competitions/{game_id}/competitors/{team_id}/statistics"
        try:
            data = _get(url)
        except Exception:
            return {}
        stats = {}
        for cat in data.get("splits", {}).get("categories", []):
            for s in cat.get("stats", []):
                stats[s["name"]] = s.get("value", 0)
        return stats

    def _fetch_nwsl(year_months, cache_path):
        """NWSL: Core stats API works — gives fouls, yellows, reds."""
        cache = Path(cache_path)
        if cache.exists():
            print(f"  Cache hit: {cache}")
            return pd.read_csv(cache, parse_dates=["date"])

        games = _get_game_index("usa.nwsl", year_months)
        print(f"  {len(games)} NWSL games. Fetching stats …")
        rows = []
        for i, g in enumerate(games):
            if i % 50 == 0:
                print(f"    {i}/{len(games)} …")
            total = {k: 0.0 for k in ["foulsCommitted", "yellowCards", "redCards"]}
            for tid in g["team_ids"]:
                s = _get_team_stats("usa.nwsl", g["game_id"], tid)
                for k in total:
                    total[k] += float(s.get(k, 0) or 0)
            rows.append({"date": g["date"], "season_year": g["season_year"],
                         "game_id": g["game_id"], **total})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["season"] = df["season_year"]
        df = df.sort_values(["season", "date"]).reset_index(drop=True)
        df["matchweek"] = df.groupby("season")["date"].transform(
            lambda x: pd.factorize(x.dt.to_period("W"))[0] + 1
        )
        df = df.rename(columns={"foulsCommitted": "total_fouls",
                                 "yellowCards": "total_yellows",
                                 "redCards": "total_reds"})
        df["total_pks"] = 0.0  # filled later from keyEvents cell
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"  Saved: {cache}")
        return df

    def _fetch_wsl(year_months, cache_path):
        """
        WSL: Core stats API is empty for older seasons.
        Use keyEvents instead — counts Yellow Card, Red Card, Penalty - Scored
        per game. Works back to 2021-22.
        """
        cache = Path(cache_path)
        if cache.exists():
            print(f"  Cache hit: {cache}")
            return pd.read_csv(cache, parse_dates=["date"])

        games = _get_game_index("eng.w.1", year_months)
        print(f"  {len(games)} WSL games. Fetching keyEvents …")
        rows = []
        for i, g in enumerate(games):
            if i % 50 == 0:
                print(f"    {i}/{len(games)} …")
            try:
                data = _get(
                    "https://site.web.api.espn.com/apis/site/v2/sports/soccer/eng.w.1/summary",
                    params={"event": g["game_id"]},
                )
                evs = data.get("keyEvents", [])
            except Exception:
                evs = []
            rows.append({
                "date": g["date"],
                "season_year": g["season_year"],
                "game_id": g["game_id"],
                "total_yellows": sum(1 for e in evs if e.get("type", {}).get("text") == "Yellow Card"),
                "total_reds":    sum(1 for e in evs if e.get("type", {}).get("text") == "Red Card"),
                "total_pks":     sum(1 for e in evs if e.get("type", {}).get("text") == "Penalty - Scored"),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        # WSL season_year from ESPN is end-year (2022 = 2021-22) → normalise to start-year
        df["season"] = df["season_year"] - 1
        df = df.sort_values(["season", "date"]).reset_index(drop=True)
        df["matchweek"] = df.groupby("season")["date"].transform(
            lambda x: pd.factorize(x.dt.to_period("W"))[0] + 1
        )
        df["total_fouls"] = float("nan")  # not available via keyEvents
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"  Saved: {cache}")
        return df

    # ── Fetch both leagues ─────────────────────────────────────────────────
    _d = Path("data")

    _nwsl_months = [f"{y}{m:02d}" for y in [2021, 2022, 2023, 2024]
                    for m in range(3, 12)]
    _wsl_months = (
        [f"2021{m:02d}" for m in range(9, 13)] +
        [f"2022{m:02d}" for m in list(range(1, 6)) + list(range(9, 13))] +
        [f"2023{m:02d}" for m in list(range(1, 6)) + list(range(9, 13))] +
        [f"2024{m:02d}" for m in list(range(1, 6)) + list(range(9, 13))] +
        [f"2025{m:02d}" for m in range(1, 6)]
    )

    print("NWSL …")
    nwsl_matches = _fetch_nwsl(_nwsl_months, cache_path=_d / "nwsl_espn.csv")
    print("WSL …")
    wsl_matches = _fetch_wsl(_wsl_months, cache_path=_d / "wsl_espn.csv")

    print(f"\nNWSL: {len(nwsl_matches)} games | seasons: {sorted(nwsl_matches['season'].unique())}")
    print(f"WSL:  {len(wsl_matches)} games | seasons: {sorted(wsl_matches['season'].unique())}")
    nwsl_matches.head(3)
    return nwsl_matches, wsl_matches


@app.cell
def _(mo):
    mo.md(r"""
    ## 1b. Fetch Penalty Data from ESPN Key Events

    ESPN's per-game stats API doesn't populate PK fields, but `keyEvents` contains
    `[Penalty - Scored]` entries per match. We count these per game.

    **Note:** This captures *penalties scored*, not all penalties awarded.
    Conversion rate ~75-80%, so it's a strong proxy for the true count.
    Saved/missed penalties don't appear in the events feed.

    ~600 games × 0.35 s ≈ 3.5 min first run. Cached to `data/nwsl_penalties.csv`.
    """)
    return


@app.cell
def _(nwsl_matches, pd):
    import requests as _req
    import time as _time
    from pathlib import Path as _Path

    _H = {"User-Agent": "Mozilla/5.0"}
    _cache = _Path("data/nwsl_penalties.csv")

    if _cache.exists():
        print("Cache hit:", _cache)
        _pen_df = pd.read_csv(_cache)
    else:
        _rows = []
        _ids = nwsl_matches["game_id"].astype(str).unique().tolist()
        print(f"Fetching keyEvents for {len(_ids)} games …")
        for _i, _gid in enumerate(_ids):
            if _i % 50 == 0:
                print(f"  {_i}/{len(_ids)} …")
            try:
                _r = _req.get(
                    "https://site.web.api.espn.com/apis/site/v2/sports/soccer/usa.nwsl/summary",
                    params={"event": _gid}, headers=_H, timeout=15,
                )
                _evs = _r.json().get("keyEvents", [])
                _pks = sum(
                    1 for e in _evs
                    if e.get("type", {}).get("text", "") == "Penalty - Scored"
                )
            except Exception:
                _pks = 0
            _rows.append({"game_id": int(_gid), "pks_scored": _pks})
            _time.sleep(0.35)

        _pen_df = pd.DataFrame(_rows)
        _pen_df.to_csv(_cache, index=False)
        print(f"Saved: {_cache}")

    # Output as a standalone lookup table — merged downstream in _make_ts
    nwsl_pk = _pen_df.rename(columns={"pks_scored": "total_pks"})
    print(f"Penalty games (scored ≥1): {(nwsl_pk['total_pks'] > 0).sum()} of {len(nwsl_pk)}")
    print(f"Average penalties scored per game: {nwsl_pk['total_pks'].mean():.3f}")
    return (nwsl_pk,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Build Matchweek-Level Time Series

    We aggregate from match level to matchweek level — taking the *average* fouls
    per game across all matches played in that matchweek.

    This gives ~22 observations per season and smooths out single-game noise
    while preserving the temporal structure ITS needs.

    Each league ends up with a continuous integer time index `t` running
    across all four seasons.
    """)
    return


@app.cell
def _(nwsl_matches, nwsl_pk, wsl_matches):
    # Merge penalty counts into nwsl_matches before aggregating
    _nwsl = nwsl_matches.merge(nwsl_pk, on="game_id", how="left")
    # Ensure total_pks exists (may not if nwsl_pk wasn't fetched or matched)
    if "total_pks" not in _nwsl.columns:
        _nwsl["total_pks"] = 0.0
    else:
        _nwsl["total_pks"] = _nwsl["total_pks"].fillna(0)

    def _make_ts(match_df, league_name):
        ts = (
            match_df.groupby(["season", "matchweek"])
            .agg(
                fouls_per_game=("total_fouls", "mean"),
                yellows_per_game=("total_yellows", "mean"),
                reds_per_game=("total_reds", "mean"),
                pks_per_game=("total_pks", "mean"),
                n_matches=("total_fouls", "count"),
            )
            .reset_index()
            .sort_values(["season", "matchweek"])
            .reset_index(drop=True)
        )
        ts["t"] = range(len(ts))
        ts["league"] = league_name
        return ts

    nwsl_ts = _make_ts(_nwsl, "NWSL")
    wsl_ts = _make_ts(wsl_matches, "WSL")

    print(f"NWSL time series: {len(nwsl_ts)} matchweeks")
    print(f"WSL time series:  {len(wsl_ts)} matchweeks")
    nwsl_ts[["season", "matchweek", "t", "fouls_per_game"]].head(8)
    return nwsl_ts, wsl_ts


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Intervention Variables

    VAR was introduced in NWSL matchweek 1 of the 2023 season (March 25, 2023).

    We create:
    - `post_var` — a step dummy: 0 before VAR, 1 after
    - `t_post_var` — time elapsed since VAR (0 before, counts up after)
    - `wwc_flag` — the Women's World Cup break (July–Aug 2023, ~matchweeks 10–16)
      which pulled star players and is a known confounder

    The `t_post_var` variable is what lets ITS detect a *slope change* —
    whether the trend shifted after VAR, not just the level.
    """)
    return


@app.cell
def _(nwsl_ts):
    # Find t index where 2023 season begins
    _intervention_row = nwsl_ts[nwsl_ts["season"] == 2023]
    if len(_intervention_row) == 0:
        raise ValueError("No 2023 data found in NWSL time series")

    intervention_t = int(_intervention_row["t"].min())

    nwsl_ts["post_var"] = (nwsl_ts["t"] >= intervention_t).astype(int)
    nwsl_ts["t_post_var"] = (
        (nwsl_ts["t"] - intervention_t) * nwsl_ts["post_var"]
    )

    # WWC confounder: approx matchweeks 10–16 of 2023 season
    nwsl_ts["wwc_flag"] = 0
    _wwc_mask = (
        (nwsl_ts["season"] == 2023)
        & (nwsl_ts["matchweek"].between(10, 16))
    )
    nwsl_ts.loc[_wwc_mask, "wwc_flag"] = 1

    print(f"VAR intervention at t = {intervention_t}")
    print(f"Pre-period matchweeks:  {(nwsl_ts['post_var'] == 0).sum()}")
    print(f"Post-period matchweeks: {(nwsl_ts['post_var'] == 1).sum()}")
    print(f"WWC flag observations:  {nwsl_ts['wwc_flag'].sum()}")
    return (intervention_t,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Exploratory Visualisation

    Before any model, look at the raw series. Two things to check:
    1. Is there a visible shift at the VAR date?
    2. Was there a pre-existing trend that was already heading somewhere?

    If the answer to (2) is yes, a naive before/after average will
    overstate VAR's effect — it will claim credit for a trend already in motion.
    """)
    return


@app.cell
def _(intervention_t, nwsl_ts, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    _metrics = [
        ("fouls_per_game", "Fouls per Game"),
        ("yellows_per_game", "Yellow Cards per Game"),
        ("reds_per_game", "Red Cards per Game"),
        ("pks_per_game", "Penalties per Game"),
    ]

    _colors = {"pre": "#4a90d9", "post": "#e07b3c"}

    for _ax, (_col, _title) in zip(_axes.flatten(), _metrics):
        _pre = nwsl_ts[nwsl_ts["post_var"] == 0]
        _post = nwsl_ts[nwsl_ts["post_var"] == 1]

        _ax.scatter(_pre["t"], _pre[_col], s=15, alpha=0.7,
                    color=_colors["pre"], label="Pre-VAR")
        _ax.scatter(_post["t"], _post[_col], s=15, alpha=0.7,
                    color=_colors["post"], label="Post-VAR")
        _ax.axvline(intervention_t, color="red", lw=1.5, ls="--",
                    label="VAR introduced")

        # Annotate WWC window if in post period
        _wwc = nwsl_ts[nwsl_ts["wwc_flag"] == 1]
        if len(_wwc) > 0:
            _ax.axvspan(_wwc["t"].min(), _wwc["t"].max(),
                        alpha=0.12, color="purple", label="WWC break")

        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _ax.legend(fontsize=7)
        _ax.grid(alpha=0.3)

    _fig.text(0.5, 0, "Matchweek (time index)", ha="center", fontsize=10)
    _fig.suptitle(
        "NWSL: Four Outcomes Before and After VAR Introduction",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Why Not Just Compare Before and After?

    Here's the naive approach: average foul rate in 2021–2022, average foul rate
    in 2023–2024, subtract. If the number is negative, VAR reduced fouls. Done.

    **The problem:** 2023 also brought:
    - Two new expansion teams with unknown baseline foul rates
    - The Women's World Cup pulling star players mid-season (July–August)
    - A new league commissioner and shifting tactical culture

    A before/after average can't untangle these. It answers *"did something change?"*
    but not *"did VAR cause it?"*

    **What ITS adds — two things:**

    1. It checks that the change happened **at the right time** — right when VAR arrived,
       not drifting in gradually across the whole year.

    2. It asks whether the **pre-period trend** was already heading somewhere.
       If foul rates were already declining before VAR, some of that drop isn't VAR's
       doing — it's a trend already in motion. Before/after misses this entirely.

    Every method below is a different way of building the **counterfactual**:
    *"What would foul rates have looked like if VAR had never arrived?"*
    They differ in what assumptions they make and how honest they are about uncertainty.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What Is Interrupted Time Series?

    Interrupted Time Series is exactly what it sounds like. You have a time series —
    a sequence of measurements over time. Something **interrupts** it at a known point.
    You want to know: did the interruption *cause* a change, or would the series
    have changed anyway?

    The key insight is that **you only ever observe one world**. VAR happened.
    You can't run a parallel NWSL without VAR and compare. So ITS builds the
    counterfactual — the "what would have happened" — from the data you have.

    **The setup here:**
    - **Outcome:** fouls, penalties, yellow cards, red cards per game (four series)
    - **Intervention:** VAR introduced, March 25, 2023
    - **Pre-period:** 2021 + 2022 seasons (~44 matchweeks of baseline)
    - **Post-period:** 2023 + 2024 seasons (~44 matchweeks post-intervention)
    - **Control series** (Method 3 only): WSL — same sport, no VAR, matched by
      matchweek within season

    Two things we check at every stage:
    - **Level shift:** did the series jump at the intervention?
    - **Slope change:** did the *trend* change after the intervention?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Method 1 — OLS Segmented Regression

    ### The Barista Analogy

    You're a regular at a coffee shop and you time your order every visit.
    Over months you see a clear pattern — it's getting slower as the shop gets busier.
    Then they hire a new barista.

    OLS draws the best straight line through your wait times *before* the new hire.
    Then draws a new straight line *after*. The gap between where the first line
    **was heading** and where the second line **actually landed** — that's the
    new barista's impact. Crucially, it accounts for the pre-existing slowdown trend
    rather than just comparing December to March averages.

    ### What it does

    Fits a linear regression:

    ```
    outcome = β₀ + β₁·t + β₂·post_var + β₃·t_post_var + β₄·wwc_flag + ε
    ```

    - `β₁`: the pre-existing trend (slope before VAR)
    - `β₂`: the **level shift** at the intervention — did fouls jump or drop the moment VAR arrived?
    - `β₃`: the **slope change** after the intervention — did the trend accelerate or reverse?
    - `β₄`: controls for the WWC player-absence confounder

    The counterfactual is the regression line with `post_var = 0` and `t_post_var = 0`
    extended forward — *"where were we heading without VAR?"*

    ### Honest limitation

    OLS assumes each matchweek is independent of the last after accounting for trend.
    Sports data violates this — if a matchweek was foul-heavy, the next one tends to be too.
    We patch this with **Newey-West HAC standard errors** (heteroskedasticity- and
    autocorrelation-consistent), but patching isn't the same as modelling.
    That's what ARIMA is for.
    """)
    return


@app.cell
def _(mo, nwsl_ts):
    import statsmodels.api as sm

    METRICS = [
        ("fouls_per_game",   "Fouls per Game"),
        ("yellows_per_game", "Yellow Cards per Game"),
        ("reds_per_game",    "Red Cards per Game"),
        ("pks_per_game",     "Penalties Scored per Game"),
    ]

    ols_results = {}

    for _metric, _label in METRICS:
        _X = sm.add_constant(
            nwsl_ts[["t", "post_var", "t_post_var", "wwc_flag"]]
        )
        _y = nwsl_ts[_metric]

        _model = sm.OLS(_y, _X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4}
        )

        # Counterfactual: what if VAR never happened?
        _X_cf = _X.copy()
        _X_cf["post_var"] = 0
        _X_cf["t_post_var"] = 0

        ols_results[_metric] = {
            "model": _model,
            "fitted": _model.fittedvalues,
            "counterfactual": _model.predict(_X_cf),
            "level_shift": _model.params["post_var"],
            "level_ci": _model.conf_int(alpha=0.05).loc["post_var"].tolist(),
            "slope_change": _model.params["t_post_var"],
            "slope_ci": _model.conf_int(alpha=0.05).loc["t_post_var"].tolist(),
            "label": _label,
        }

    # ── Print coefficient table ────────────────────────────────────────────
    _rows = []
    for _metric, _label in METRICS:
        _r = ols_results[_metric]
        _rows.append({
            "Outcome": _label,
            "Level shift (β₂)": f"{_r['level_shift']:+.3f}",
            "95% CI": f"[{_r['level_ci'][0]:+.3f}, {_r['level_ci'][1]:+.3f}]",
            "Slope Δ (β₃)": f"{_r['slope_change']:+.4f}",
            "Slope 95% CI": f"[{_r['slope_ci'][0]:+.4f}, {_r['slope_ci'][1]:+.4f}]",
        })

    mo.callout(
        mo.md("**OLS coefficients with Newey-West HAC standard errors**"),
        kind="info",
    )
    return METRICS, ols_results, sm


@app.cell
def _(METRICS, intervention_t, nwsl_ts, ols_results, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for _ax, (_metric, _label) in zip(_axes.flatten(), METRICS):
        _r = ols_results[_metric]
        _t = nwsl_ts["t"]
        _y = nwsl_ts[_metric]

        # Raw data
        _ax.scatter(_t, _y, s=12, alpha=0.5, color="#888", zorder=2)

        # OLS fitted line
        _ax.plot(_t, _r["fitted"], color="#2c7bb6", lw=1.8,
                 label="OLS fitted", zorder=3)

        # Counterfactual (post-period only)
        _post_mask = nwsl_ts["post_var"] == 1
        _ax.plot(
            _t[_post_mask], _r["counterfactual"][_post_mask],
            color="#d7191c", lw=1.8, ls="--",
            label="Counterfactual (no VAR)", zorder=3,
        )

        # Shade effect gap
        _ax.fill_between(
            _t[_post_mask],
            _r["counterfactual"][_post_mask],
            _r["fitted"][_post_mask],
            alpha=0.15, color="#2c7bb6",
        )

        # Intervention line
        _ax.axvline(intervention_t, color="red", lw=1.5, ls="--")

        _ax.set_title(_label, fontsize=10, fontweight="bold")
        _ax.legend(fontsize=7)
        _ax.grid(alpha=0.3)

    _fig.text(0.5, 0, "Matchweek (time index)", ha="center", fontsize=10)
    _fig.suptitle(
        "Method 1: OLS Segmented Regression — Fitted vs Counterfactual",
        fontsize=12, fontweight="bold", y=1.01,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## What Is Autocorrelation?

    Think about your own energy levels. If you had a terrible sleep last night,
    you're probably tired today. If you're tired today, there's a decent chance
    you'll be tired tomorrow too — you haven't caught up yet.

    That's **autocorrelation**: today's value is correlated with yesterday's value.
    The past bleeds into the present.

    In NWSL data, if a matchweek had a lot of aggression — maybe a run of
    high-stakes fixtures — that intensity can carry forward. Or a foul-heavy
    refereeing crew gets assigned to the next round. The series has **memory**.

    **Why it matters for ITS:**

    OLS assumes each observation is independent. When a series has memory,
    OLS underestimates the uncertainty in its estimates — the standard errors
    are too small, and you can end up confident about things you shouldn't be.

    Newey-West HAC (Method 1) patches this after the fact.
    ARIMA (Method 2) *models* the memory structure before measuring anything.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Method 2 — ARIMA with Intervention

    ### The Commute Analogy

    Your morning commute isn't random day to day. If there was a crash Monday,
    Tuesday is still rough — cones are out, drivers cautious. Wednesday starts
    recovering. **The past bleeds into the present.**

    ARIMA maps out exactly how much yesterday predicts today before measuring
    anything. Then it measures the intervention *on top of* that corrected baseline.
    The noise model is built in, not patched.

    ### What it does

    Two stages:

    **Stage 1 — Model selection on pre-period only:**
    Use `auto_arima` on the pre-VAR data to find the best (p, d, q) order.
    - `p`: how many past values predict today (autoregressive terms)
    - `d`: how many times we difference the series to make it stationary
    - `q`: how many past *errors* predict today (moving-average terms)

    **Stage 2 — Refit full series with intervention as exogenous variable:**
    Pass `post_var`, `t_post_var`, and `wwc_flag` as external regressors to SARIMAX.
    The intervention coefficient is your causal estimate, now with a properly
    modelled noise structure.

    ### Honest limitation

    The (p, d, q) order is a researcher choice — different selection criteria
    (AIC vs BIC) can give different orders. The shape of the intervention variable
    (step function, ramp, pulse) also matters. Both choices affect your estimate
    and should be reported.
    """)
    return


@app.cell
def _(METRICS, mo, nwsl_ts):
    try:
        from pmdarima import auto_arima as _auto_arima
        _pmdarima_ok = True
    except ImportError:
        _pmdarima_ok = False

    mo.stop(
        not _pmdarima_ok,
        mo.callout(
            mo.md("**`pmdarima` not installed.** Run `pip install pmdarima` then restart."),
            kind="danger",
        ),
    )

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    arima_results = {}
    arima_orders = {}

    for _metric, _label in METRICS:
        # Stage 1: auto_arima on pre-period only
        _pre = nwsl_ts[nwsl_ts["post_var"] == 0][_metric].values
        _search = _auto_arima(
            _pre,
            seasonal=False,
            stepwise=False,          # exhaustive search, not stepwise
            information_criterion="aicc",  # corrected AIC for small n (~44 obs)
            max_p=4, max_q=2,
            suppress_warnings=True,
            error_action="ignore",
        )
        _p, _d, _q = _search.order
        # Fallback: (0,0,0) is a white-noise model — if ACF shows persistence,
        # force AR(1) rather than pretending observations are independent.
        if (_p, _d, _q) == (0, 0, 0):
            _p = 1
        arima_orders[_metric] = (_p, _d, _q)

        # Stage 2: refit full series with intervention exogenous variables
        _exog = nwsl_ts[["post_var", "t_post_var", "wwc_flag"]].values
        _model = SARIMAX(
            nwsl_ts[_metric],
            order=(_p, _d, _q),
            exog=_exog,
        ).fit(disp=False)

        # Counterfactual: zero out intervention variables
        _exog_cf = _exog.copy()
        _exog_cf[:, 0] = 0   # post_var = 0
        _exog_cf[:, 1] = 0   # t_post_var = 0

        arima_results[_metric] = {
            "model": _model,
            "order": (_p, _d, _q),
            "fitted": _model.fittedvalues,
            "intervention_coef": _model.params.get(
                "post_var", _model.params.iloc[-3]
            ),
            "intervention_ci": _model.conf_int().iloc[-3].tolist()
            if "post_var" not in _model.params
            else _model.conf_int().loc["post_var"].tolist(),
            "label": _label,
        }

        print(f"{_label}: ARIMA{(_p,_d,_q)}, "
              f"intervention coef = {arima_results[_metric]['intervention_coef']:+.3f}")
    return (arima_results,)


@app.cell
def _(METRICS, arima_results, intervention_t, nwsl_ts, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for _ax, (_metric, _label) in zip(_axes.flatten(), METRICS):
        _r = arima_results[_metric]
        _t = nwsl_ts["t"]

        _ax.scatter(_t, nwsl_ts[_metric], s=12, alpha=0.5,
                    color="#888", zorder=2, label="Observed")
        _ax.plot(_t, _r["fitted"], color="#1a9641", lw=1.8,
                 label=f"ARIMA{_r['order']} fitted", zorder=3)
        _ax.axvline(intervention_t, color="red", lw=1.5, ls="--",
                    label="VAR introduced")

        _ax.set_title(
            f"{_label}  |  ARIMA{_r['order']}",
            fontsize=10, fontweight="bold",
        )
        _ax.legend(fontsize=7)
        _ax.grid(alpha=0.3)

    _fig.text(0.5, 0, "Matchweek (time index)", ha="center", fontsize=10)
    _fig.suptitle(
        "Method 2: ARIMA with Intervention — Fitted Values",
        fontsize=12, fontweight="bold", y=1.01,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(METRICS, arima_results, nwsl_ts, plt, sm):
    # ── ACF of OLS residuals vs ARIMA residuals ────────────────────────────
    # Shows that ARIMA removed the autocorrelation structure OLS left behind

    _fig, _axes = plt.subplots(len(METRICS), 2, figsize=(12, 10))

    for _i, (_metric, _label) in enumerate(METRICS):
        _X = sm.add_constant(
            nwsl_ts[["t", "post_var", "t_post_var", "wwc_flag"]]
        )
        _ols_resid = sm.OLS(nwsl_ts[_metric], _X).fit().resid
        _arima_resid = arima_results[_metric]["model"].resid

        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(_ols_resid, ax=_axes[_i, 0], lags=15, alpha=0.05)
        _axes[_i, 0].set_title(f"OLS residuals — {_label}", fontsize=9)
        _axes[_i, 0].set_ylim(-1, 1)

        plot_acf(_arima_resid.dropna(), ax=_axes[_i, 1], lags=15, alpha=0.05)
        _axes[_i, 1].set_title(f"ARIMA residuals — {_label}", fontsize=9)
        _axes[_i, 1].set_ylim(-1, 1)

    _fig.suptitle(
        "ACF of Residuals: OLS (left) vs ARIMA (right)\n"
        "Bars outside the blue band = remaining autocorrelation",
        fontsize=11, fontweight="bold",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Method 3 — CausalImpact (Bayesian Structural Time Series)

    ### The Housemate Analogy

    You decide to start going to bed an hour earlier and want to know if it
    genuinely improved your energy. The problem: energy changes with seasons,
    stress, work. How do you separate the sleep change from everything else?

    Now imagine your housemate doesn't change their sleep at all. You've both
    been tracking energy for months and they tend to move together — when work
    is stressful, you're both tired.

    CausalImpact watches that shared pattern *before* your bedtime change,
    learns it deeply, then asks: **given how your housemate's energy is tracking
    right now, here's what yours should look like without the sleep change.**
    The gap between that prediction and your actual energy is your answer.

    **That's the WSL.** It never got VAR. It's your housemate.

    ### What it does

    1. Fits a **Bayesian state space model** on the pre-period using NWSL foul rate
       as outcome and WSL foul rate as a covariate
    2. Projects that model forward into the post-period as the counterfactual
    3. The difference between the counterfactual and observed is the causal effect

    ### What you get that OLS and ARIMA don't

    **A credible interval**, not a confidence interval. This is a real distinction:
    - Confidence interval: *"If we repeated this experiment many times, 95% of
      intervals would contain the true value."* (A statement about the procedure.)
    - Credible interval: *"Given this data, there is a 95% probability the effect
      falls between X and Y."* (A direct statement about this result.)

    For a general audience, the credible interval is easier to explain and more
    honest about what the model actually knows.

    You also get three panels automatically: observed vs. counterfactual,
    pointwise effect at each matchweek, and the **cumulative effect** — the total
    accumulated impact across the post-period.

    ### Honest limitation

    The WSL control series must not be affected by the NWSL intervention. Since
    WSL had no VAR and operates independently, this holds. But the leagues aren't
    identical — different countries, player pools, tactical cultures. The relationship
    learned pre-intervention may not be perfectly stable post-intervention.
    Worth acknowledging, not fatal.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### WSL as Control Series

    WSL keyEvents give us yellows, reds, and penalties scored per game back to
    2021-22 — the full pre-intervention window we need.

    **For fouls:** WSL keyEvents don't include individual foul events (only
    cards and goals), so the fouls model runs NWSL-only.

    **For yellows, reds, penalties:** WSL is used as a covariate. CausalImpact
    learns the NWSL–WSL co-movement in the pre-period and uses it to project
    what NWSL *would have looked like* without VAR.
    """)
    return


@app.cell
def _(intervention_t, mo, np, nwsl_ts, wsl_ts):
    try:
        from causalimpact import CausalImpact
        _ci_available = True
    except ImportError:
        _ci_available = False

    mo.stop(
        not _ci_available,
        mo.callout(
            mo.md("**`pycausalimpact` not installed.**  Run `pip install pycausalimpact` then restart."),
            kind="danger",
        ),
    )

    # Metrics where WSL covariate is available (yellows, reds, pks)
    # Fouls: NWSL-only (WSL keyEvents don't carry foul counts)
    _WSL_AVAILABLE = {"yellows_per_game", "reds_per_game", "pks_per_game"}

    ci_results = {}

    for _metric, _label in [
        ("fouls_per_game",   "Fouls per Game"),
        ("yellows_per_game", "Yellow Cards per Game"),
        ("reds_per_game",    "Red Cards per Game"),
        ("pks_per_game",     "Penalties Scored per Game"),
    ]:
        if _metric in _WSL_AVAILABLE:
            # Align WSL matchweek series as covariate
            _wsl_col = _metric
            _wsl_cov = wsl_ts[["season", "matchweek", _wsl_col]].rename(
                columns={_wsl_col: "wsl"}
            )
            _joined = nwsl_ts[["t", "season", "matchweek", _metric]].merge(
                _wsl_cov, on=["season", "matchweek"], how="inner"
            ).sort_values("t")
            _ci_df = _joined[[_metric, "wsl"]].copy()
            _ci_df.index = _joined["t"].values
        else:
            # Fouls: NWSL-only
            _ci_df = nwsl_ts[[_metric]].copy()
            _ci_df.index = nwsl_ts["t"].values

        # Validate that response has variation and no NaN
        _response = _ci_df.iloc[:, 0]
        _nan_count = _response.isna().sum()
        _var = _response.var()

        if _nan_count > 0 or _var == 0 or (isinstance(_var, float) and np.isnan(_var)):
            print(f"⚠ {_label}: Skipped (insufficient variation or all NaN). "
                  f"NaN: {_nan_count}, Variance: {_var}")
            continue

        _t_vals = sorted(_ci_df.index.tolist())
        _split = max(t for t in _t_vals if t < intervention_t)
        _post_start = intervention_t if intervention_t in _t_vals \
                      else _t_vals[_t_vals.index(_split) + 1]
        _pre  = [_t_vals[0], _split]
        _post = [_post_start, _t_vals[-1]]

        _ci = CausalImpact(
            _ci_df, _pre, _post,
            model_args={"fit_method": "hmc"},
        )

        ci_results[_metric] = {
            "ci": _ci,
            "summary": _ci.summary(),
            "report": _ci.summary(output="report"),
            "label": _label,
            "used_wsl": _metric in _WSL_AVAILABLE,
        }
    return (ci_results,)


@app.cell
def _(ci_results, plt):
    # Three-panel CausalImpact plot for each metric
    for _metric, _r in ci_results.items():
        _r["ci"].plot()
        plt.suptitle(
            f"CausalImpact: {_r['label']}",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(ci_results, mo):
    # Plain-English summaries
    _tabs = {}
    for _metric, _r in ci_results.items():
        _tabs[_r["label"]] = mo.md(
            f"**Numerical summary:**\n\n```\n{_r['summary']}\n```\n\n"
            f"**Plain-English report:**\n\n{_r['report']}"
        )

    mo.ui.tabs(_tabs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## What All Three Methods Tell Us

    The table below compares what each method gives you — not just the numbers,
    but the *type* of answer and the *assumptions* baked in.
    Where all three agree directionally, we can be confident.
    Where they diverge, that tells you which assumptions are doing the work.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | | OLS | ARIMA | CausalImpact |
    |---|---|---|---|
    | **Interval type** | Confidence (HAC-corrected) | Confidence | **Credible** |
    | **Autocorrelation** | Patched post-hoc | Built into model | Built into model |
    | **Uncertainty widens over time** | No — constant SE | Yes | Yes |
    | **Control series used** | No | No | Yes (WSL) |
    | **Cumulative effect** | No | No | Yes — in third panel |
    | **Plain-English output** | Harder | Harder | Auto-generated |
    | **Key assumption** | Linear trend, independence | Stationarity after differencing | WSL–NWSL relationship stable |

    **How to read this:** OLS is the baseline — fast and interpretable but makes
    the strongest independence assumptions. ARIMA earns its estimate by modelling
    the memory structure. CausalImpact brings in external evidence (the WSL)
    and gives the most honest uncertainty picture.

    If they agree on direction and rough magnitude: strong result.
    If they disagree: investigate which assumption is being violated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Robustness Checks

    Three checks that separate a rigorous analysis from a naive one.
    Each is one paragraph in the post; each is easy to implement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check 1 — Placebo Test

    Run the same ITS as if VAR had been introduced **in 2022** — a fake intervention date.
    You should find no significant effect.

    If you *do* find an effect at the fake date, the pre-period trend wasn't actually
    stable, which undermines everything. This is the most important check.
    """)
    return


@app.cell
def _(nwsl_ts, sm):
    # Placebo: fake VAR introduction at mid-2022
    _placebo_row = nwsl_ts[
        (nwsl_ts["season"] == 2022) & (nwsl_ts["matchweek"] == 11)
    ]
    if len(_placebo_row) == 0:
        print("No 2022 matchweek 11 — adjusting placebo to first 2022 matchweek 10")
        _placebo_row = nwsl_ts[
            (nwsl_ts["season"] == 2022) & (nwsl_ts["matchweek"] >= 10)
        ]

    _placebo_t = int(_placebo_row["t"].values[0])

    # Use only pre-2023 data
    _df = nwsl_ts[nwsl_ts["season"].isin([2021, 2022])].copy()
    _df["placebo_post"] = (_df["t"] >= _placebo_t).astype(int)
    _df["placebo_t_post"] = (_df["t"] - _placebo_t) * _df["placebo_post"]

    print(f"Placebo intervention at t = {_placebo_t}")
    print("\n── Placebo OLS (fouls_per_game) ──────────────────────────")

    _X_p = sm.add_constant(_df[["t", "placebo_post", "placebo_t_post"]])
    _y_p = _df["fouls_per_game"]
    _m_p = sm.OLS(_y_p, _X_p).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    print(f"  Placebo level shift:  {_m_p.params['placebo_post']:+.3f}  "
          f"(p = {_m_p.pvalues['placebo_post']:.3f})")
    print(f"  Placebo slope change: {_m_p.params['placebo_t_post']:+.4f}  "
          f"(p = {_m_p.pvalues['placebo_t_post']:.3f})")
    print("\nInterpretation: p-values should be large (> 0.05) "
          "— no effect at a fake intervention date is what we want.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check 2 — WWC Sensitivity

    Re-run all three methods **excluding** the WWC window (matchweeks 10–16 of 2023).
    If results hold, the World Cup player absences weren't driving the finding.
    If results collapse, the WWC was doing more work than the VAR model suggests.
    """)
    return


@app.cell
def _(METRICS, nwsl_ts, sm):
    # Exclude WWC matchweeks and rerun OLS
    _df_no_wwc = nwsl_ts[nwsl_ts["wwc_flag"] == 0].copy()

    print("── WWC Sensitivity: OLS without WWC matchweeks ──────────────")
    for _metric, _label in METRICS:
        _X = sm.add_constant(_df_no_wwc[["t", "post_var", "t_post_var"]])
        _y = _df_no_wwc[_metric]
        _m = sm.OLS(_y, _X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        _lo, _hi = _m.conf_int().loc["post_var"]
        print(f"  {_label:<28}  level shift = {_m.params['post_var']:+.3f}  "
              f"95% CI [{_lo:+.3f}, {_hi:+.3f}]  p = {_m.pvalues['post_var']:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check 3 — Compare OLS Coefficients Across All Outcomes

    Across all four outcomes (fouls, yellows, reds, PKs), are the directions
    consistent with what VAR *should* do?

    Expected directions from prior literature:
    - **Fouls:** decrease — deterrence effect (players know box fouls will be caught)
    - **Penalties:** increase — VAR reviews more box incidents
    - **Yellow cards:** ambiguous — VAR can add or rescind
    - **Red cards:** ambiguous — low frequency, likely noisy

    If fouls go up and penalties go down, something is wrong with the model
    or the data, not just the theory.
    """)
    return


@app.cell
def _(METRICS, mo, ols_results, pd):
    _rows = []
    for _metric, _label in METRICS:
        _r = ols_results[_metric]
        _lo, _hi = _r["level_ci"]
        _sig = "✓" if not (_lo < 0 < _hi) else "—"
        _rows.append({
            "Outcome": _label,
            "Level shift": f"{_r['level_shift']:+.3f}",
            "95% CI": f"[{_lo:+.3f}, {_hi:+.3f}]",
            "Significant?": _sig,
            "Expected direction": {
                "fouls_per_game": "↓ (deterrence)",
                "yellows_per_game": "ambiguous",
                "reds_per_game": "ambiguous",
                "pks_per_game": "↑ (VAR detects box fouls)",
            }[_metric],
        })

    mo.ui.table(pd.DataFrame(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Honest Limitations

    Naming these is what separates a rigorous post from a naive one.
    Your audience will trust you more for it.

    1. **Short pre-period** — only 2 clean seasons before VAR. The pre-trend
       estimate is less stable than you'd want. More data would help.

    2. **2023 confounders** — expansion teams, WWC break, new commissioner
       all co-occur with VAR. The WWC flag helps but doesn't fully solve this.

    3. **WSL comparability** — different country, player pool, tactical culture.
       The control series is reasonable but imperfect.

    4. **Foul type granularity** — FBref doesn't break out handball fouls, box
       fouls, or tactical fouls separately. Penalty and card series are proxies,
       not direct VAR-outcome measures.

    5. **Expansion teams** — 2023 had new teams whose baseline foul rates are
       unknown. This adds noise to the post-intervention series.
    """)
    return


if __name__ == "__main__":
    app.run()
