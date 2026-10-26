"""ML pipeline for flight price prediction.

Steps:
  1. prepare_data      - clean & unify raw columns, engineer features
  2. split_and_cv      - hold-out test split + cross-validation schemes
  3. regression        - linear (Ridge) baseline
  4. ensemble models   - Random Forest (tuned) + Gradient Boosting, model selection
     nn_export         - NN-ready (encoded + standardised) data with the same split / folds
  6. predict_calendar  - per-day forecast for a route / year / months-in-advance
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_COLUMNS = ["scrape_date", "flight_date", "origin", "dest", "dep_time",
               "duration", "price_raw", "airline"]

# Fixed-date public holidays (CZ + common European) used for holiday features.
HOLIDAYS_MD = [(1, 1), (1, 6), (5, 1), (5, 8), (7, 5), (7, 6), (9, 28),
               (10, 28), (11, 17), (12, 24), (12, 25), (12, 26), (12, 31)]

# Ordinal encoding of the departure part-of-day.
DEP_PERIODS = ["night", "morning", "afternoon", "evening"]

NOMINAL = ["dest", "airline", "dep_period"]
NUMERIC = ["days_before", "dom", "dow", "is_weekend", "is_holiday",
           "holiday_dist", "dep_minutes", "duration_min", "dep_period_ord"]
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Step 1 - data preparation
# --------------------------------------------------------------------------- #
@dataclass
class PrepReport:
    n_raw: int = 0
    n_clean: int = 0
    fixes: dict = field(default_factory=dict)
    raw_sample: pd.DataFrame | None = None


def _parse_price(s: str) -> float:
    m = re.search(r"(\d[\d,]*\.?\d*)", str(s))
    return float(m.group(1).replace(",", "")) if m else np.nan


def _parse_duration(s: str) -> float:
    s = str(s)
    h = re.search(r"(\d+)\s*h", s)
    m = re.search(r"(\d+)\s*min", s)
    if not h and not m:
        return np.nan
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)


def _dep_period(minutes: float) -> str:
    if minutes < 6 * 60:
        return "night"
    if minutes < 12 * 60:
        return "morning"
    if minutes < 18 * 60:
        return "afternoon"
    return "evening"


def holiday_features(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (is_holiday, distance in days to the nearest fixed holiday, capped at 15)."""
    dates = pd.to_datetime(dates)
    dist = np.full(len(dates), 15, dtype=float)
    for y_off in (-1, 0, 1):
        for mth, d in HOLIDAYS_MD:
            hol = pd.to_datetime(dict(year=dates.dt.year + y_off, month=mth, day=d))
            dist = np.minimum(dist, (dates - hol).dt.days.abs().to_numpy())
    dist = pd.Series(dist, index=dates.index)
    return (dist == 0).astype(int), dist


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    fd = pd.to_datetime(df["flight_date"])
    df["dom"] = fd.dt.day
    df["dow"] = fd.dt.dayofweek                      # ordinal 0=Mon .. 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"], df["holiday_dist"] = holiday_features(fd)
    return df


def load_raw(source) -> pd.DataFrame:
    """source: path or raw bytes of a CSV file (with or without a header)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.StringIO(source.decode("utf-8", errors="replace"))
    df = pd.read_csv(source, header=None, dtype=str, keep_default_na=False)
    # Drop a header row if present.
    if not re.match(r"\d", str(df.iloc[0, 0])):
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :8]
    while df.shape[1] < 8:
        df[df.shape[1]] = ""
    df.columns = RAW_COLUMNS
    return df


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, PrepReport]:
    rep = PrepReport(n_raw=len(raw), raw_sample=raw.head(8).copy())
    df = raw.copy()
    for c in RAW_COLUMNS:
        df[c] = df[c].astype(str).str.strip()

    # Fix rows where the price and airline columns were merged ("US$36.92Wizz Air").
    merged = df["airline"].eq("") & df["price_raw"].str.contains(r"\d[A-Za-z]")
    split = df.loc[merged, "price_raw"].str.extract(r"^(.*?\d+(?:\.\d+)?)(\D.*)$")
    df.loc[merged, "price_raw"] = split[0]
    df.loc[merged, "airline"] = split[1].str.strip()
    rep.fixes["Merged price/airline cells split"] = int(merged.sum())

    # Dates: corrupted scrape dates (e.g. '20102-110-01') are repaired by forward
    # filling from the neighbouring row - the file is ordered by scrape date.
    scrape = pd.to_datetime(df["scrape_date"], format="%Y-%m-%d", errors="coerce")
    rep.fixes["Corrupted scrape dates repaired (forward fill)"] = int(scrape.isna().sum())
    df["scrape_date"] = scrape.ffill().bfill()
    flight = pd.to_datetime(df["flight_date"], format="%Y-%m-%d", errors="coerce")
    rep.fixes["Unparseable flight dates dropped"] = int(flight.isna().sum())
    df["flight_date"] = flight
    df = df[flight.notna()]

    # Price: unify 'US$59.47' / '$59.47' -> 59.47 (USD).
    df["price"] = df["price_raw"].map(_parse_price)
    rep.fixes["Price formats unified to numeric USD"] = int((~df["price_raw"].str.startswith("US$")).sum())

    # Departure time -> minutes after midnight + ordinal part-of-day.
    # Times come both as '09:20 AM' (12h) and '17:05' (24h).
    t12 = pd.to_datetime(df["dep_time"], format="%I:%M %p", errors="coerce")
    t = t12.fillna(pd.to_datetime(df["dep_time"], format="%H:%M", errors="coerce"))
    rep.fixes["24h departure times unified with 12h AM/PM format"] = int((t12.isna() & t.notna()).sum())
    df["dep_minutes"] = t.dt.hour * 60 + t.dt.minute
    df["duration_min"] = df["duration"].map(_parse_duration)

    # Nominal columns: normalise text, group rare airlines (<0.5%) into "Other".
    df["airline"] = df["airline"].replace("", "Unknown").str.title()
    df["dest"] = df["dest"].str.upper()
    df["origin"] = df["origin"].str.upper()
    share = df["airline"].value_counts(normalize=True)
    rare = share[share < 0.005].index
    rep.fixes["Rare airlines grouped into 'Other'"] = int(df["airline"].isin(rare).sum())
    df.loc[df["airline"].isin(rare), "airline"] = "Other"

    before = len(df)
    df = df.dropna(subset=["price", "dep_minutes", "duration_min"])
    rep.fixes["Rows with missing numeric values dropped"] = before - len(df)

    df["days_before"] = (df["flight_date"] - df["scrape_date"]).dt.days
    before = len(df)
    df = df[df["days_before"] >= 0]
    rep.fixes["Rows scraped after departure dropped"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["scrape_date", "flight_date", "origin", "dest", "dep_time", "airline"])
    rep.fixes["Duplicate rows dropped"] = before - len(df)

    # Outliers: remove extreme prices per route (log-IQR rule, k=3).
    lp = np.log(df["price"])
    q1 = lp.groupby(df["dest"]).transform(lambda s: s.quantile(0.25))
    q3 = lp.groupby(df["dest"]).transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    keep = (lp >= q1 - 3 * iqr) & (lp <= q3 + 3 * iqr) & (df["price"] > 0)
    rep.fixes["Extreme price outliers removed (log-IQR, k=3)"] = int((~keep).sum())
    df = df[keep]

    df["route"] = df["origin"] + "-" + df["dest"]
    df["dep_period"] = df["dep_minutes"].map(_dep_period)
    df["dep_period_ord"] = df["dep_period"].map({p: i for i, p in enumerate(DEP_PERIODS)})
    df = calendar_features(df)
    df["log_price"] = np.log1p(df["price"])
    df = df.sort_values(["scrape_date", "flight_date", "dest"]).reset_index(drop=True)
    rep.n_clean = len(df)
    return df, rep


def column_types_table() -> pd.DataFrame:
    rows = [
        ("scrape_date", "date", "-> days_before (ordinal/numeric lead time)"),
        ("flight_date", "date", "-> dom, dow (ordinal), is_weekend, is_holiday, holiday_dist"),
        ("origin / dest", "nominal", "route key; dest one-hot encoded"),
        ("dep_time", "ordinal (time)", "12h/24h unified -> dep_minutes + dep_period (night<morning<afternoon<evening)"),
        ("duration", "ratio", "'1h 45min' -> 105 minutes"),
        ("price", "ratio (target)", "'US$59.47' / '$59.47' -> 59.47, modelled as log(1+price)"),
        ("airline", "nominal", "title-cased, rare (<0.5%) -> 'Other', one-hot encoded"),
    ]
    return pd.DataFrame(rows, columns=["Column", "Scale", "Transformation"])


# --------------------------------------------------------------------------- #
# Step 2 - train/test split + cross-validation
# --------------------------------------------------------------------------- #
def make_preprocessor(linear: bool) -> ColumnTransformer:
    if linear:
        # Linear model: day-of-week as one-hot, numeric features standardised,
        # quadratic lead-time term to capture the booking curve.
        return ColumnTransformer([
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL + ["dow"]),
            ("num", StandardScaler(), ["days_before", "days_before_sq", "dom", "is_weekend",
                                       "is_holiday", "holiday_dist", "dep_minutes", "duration_min"]),
        ])
    return ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
        ("num", "passthrough", NUMERIC),
    ])


def add_model_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["days_before_sq"] = df["days_before"] ** 2
    return df


@dataclass
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    cv_schemes: dict          # name -> list of (train_idx, val_idx) relative to train set
    cv_scores: dict           # name -> list of fold MAE (USD) for the Ridge baseline


def split_and_cv(df: pd.DataFrame, n_splits: int = 5) -> SplitResult:
    # Hold-out: 20 % of flight dates (all their observations) never seen in training.
    groups = df["flight_date"].dt.strftime("%Y-%m-%d") + df["dest"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(df, groups=groups))
    train = df.iloc[tr]
    g_train = groups.iloc[tr]

    schemes = {
        "K-Fold (random rows)": list(KFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(train)),
        "Group K-Fold (by flight day)": list(GroupKFold(n_splits).split(train, groups=g_train)),
    }
    # Time-series split on scrape date: train on earlier observations, validate on later.
    order = np.argsort(train["scrape_date"].to_numpy(), kind="stable")
    schemes["Time-Series (by scrape date)"] = [(order[a], order[b])
                                               for a, b in TimeSeriesSplit(n_splits).split(order)]

    scores = {}
    X = add_model_cols(train)
    for name, folds in schemes.items():
        maes = []
        for a, b in folds:
            m = Pipeline([("prep", make_preprocessor(True)), ("model", Ridge(alpha=1.0))])
            m.fit(X.iloc[a], X["log_price"].iloc[a])
            pred = np.expm1(m.predict(X.iloc[b]))
            maes.append(mean_absolute_error(X["price"].iloc[b], pred))
        scores[name] = maes
    return SplitResult(tr, te, schemes, scores)


# --------------------------------------------------------------------------- #
# Steps 3 & 4 - models
# --------------------------------------------------------------------------- #
def _metrics(y, p) -> dict:
    return {"MAE": mean_absolute_error(y, p),
            "MAPE": mean_absolute_percentage_error(y, p) * 100,
            "R2": r2_score(y, p)}


def _cv_mae(make_model, X, folds) -> list[float]:
    out = []
    for a, b in folds:
        m = make_model()
        m.fit(X.iloc[a], X["log_price"].iloc[a])
        out.append(mean_absolute_error(X["price"].iloc[b], np.expm1(m.predict(X.iloc[b]))))
    return out


@dataclass
class ModelResults:
    models: dict              # name -> fitted pipeline (fitted on full train set)
    cv: dict                  # name -> list of fold MAE
    test: dict                # name -> metrics dict
    test_pred: dict           # name -> np.ndarray predictions on test set
    ridge_coefs: pd.Series
    rf_grid: pd.DataFrame
    importances: pd.Series
    best: str
    checks: list              # (requirement, passed, detail)


def train_models(df: pd.DataFrame, split: SplitResult, progress=lambda msg: None) -> ModelResults:
    X = add_model_cols(df)
    train, test = X.iloc[split.train_idx], X.iloc[split.test_idx]
    folds = split.cv_schemes["Group K-Fold (by flight day)"]
    y_test = test["price"].to_numpy()

    def ridge():
        return Pipeline([("prep", make_preprocessor(True)), ("model", Ridge(alpha=1.0))])

    def rf(depth=None, leaf=1, n=150):
        return Pipeline([("prep", make_preprocessor(False)),
                         ("model", RandomForestRegressor(n_estimators=n, max_depth=depth,
                                                         min_samples_leaf=leaf, max_features=0.5,
                                                         n_jobs=-1, random_state=RANDOM_STATE))])

    def hgb():
        return Pipeline([("prep", make_preprocessor(False)),
                         ("model", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                                 max_leaf_nodes=31,
                                                                 random_state=RANDOM_STATE))])

    cv, models, test_m, test_p = {}, {}, {}, {}

    # Step 3: linear regression baseline.
    progress("Step 3: Ridge regression")
    cv["Ridge regression"] = _cv_mae(ridge, train, folds)
    models["Ridge regression"] = ridge().fit(train, train["log_price"])

    # Step 4a: Random Forest hyper-parameter grid (3 folds to keep it quick).
    progress("Step 4: Random Forest grid search")
    grid_rows = []
    small_folds = folds[:3]
    for depth in (8, 16, None):
        for leaf in (1, 5, 15):
            maes = _cv_mae(lambda: rf(depth, leaf, n=80), train, small_folds)
            grid_rows.append({"max_depth": str(depth), "min_samples_leaf": leaf, "CV MAE": np.mean(maes)})
    rf_grid = pd.DataFrame(grid_rows)
    best_row = rf_grid.loc[rf_grid["CV MAE"].idxmin()]
    best_depth = None if best_row["max_depth"] == "None" else int(best_row["max_depth"])
    best_leaf = int(best_row["min_samples_leaf"])

    progress("Step 4: Random Forest (tuned) cross-validation")
    cv["Random Forest"] = _cv_mae(lambda: rf(best_depth, best_leaf), train, folds)
    models["Random Forest"] = rf(best_depth, best_leaf, n=250).fit(train, train["log_price"])

    progress("Step 4: Gradient Boosting cross-validation")
    cv["Gradient Boosting"] = _cv_mae(hgb, train, folds)
    models["Gradient Boosting"] = hgb().fit(train, train["log_price"])

    for name, m in models.items():
        test_p[name] = np.expm1(m.predict(test))
        test_m[name] = _metrics(y_test, test_p[name])

    # Naive baseline: mean training price per route and lead-week.
    key_tr = train["dest"] + "_" + (train["days_before"] // 7).astype(str)
    key_te = test["dest"] + "_" + (test["days_before"] // 7).astype(str)
    naive = key_te.map(train.groupby(key_tr)["price"].mean()).fillna(train["price"].mean()).to_numpy()
    test_m["Naive (route x lead-week mean)"] = _metrics(y_test, naive)
    test_p["Naive (route x lead-week mean)"] = naive

    best = min(cv, key=lambda k: np.mean(cv[k]))

    # Ridge coefficients.
    rp = models["Ridge regression"]
    coefs = pd.Series(rp.named_steps["model"].coef_,
                      index=rp.named_steps["prep"].get_feature_names_out()).sort_values()

    # RF importances aggregated back to the original columns.
    rfp = models["Random Forest"]
    names = rfp.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rfp.named_steps["model"].feature_importances_, index=names)
    agg = {}
    for n, v in imp.items():
        base = n.split("__", 1)[1]
        for col in NOMINAL:
            if base.startswith(col + "_"):
                base = col
                break
        agg[base] = agg.get(base, 0) + v
    importances = pd.Series(agg).sort_values()

    tb = test_m[best]
    checks = [
        ("Best model beats the linear regression baseline (CV MAE)",
         np.mean(cv[best]) <= np.mean(cv["Ridge regression"]),
         f"{np.mean(cv[best]):.2f} vs {np.mean(cv['Ridge regression']):.2f} USD"),
        ("Best model beats the naive route x lead-week average (test MAE)",
         tb["MAE"] < test_m["Naive (route x lead-week mean)"]["MAE"],
         f"{tb['MAE']:.2f} vs {test_m['Naive (route x lead-week mean)']['MAE']:.2f} USD"),
        ("Generalises to unseen flight days: test MAPE < 25 %", tb["MAPE"] < 25, f"MAPE = {tb['MAPE']:.1f} %"),
        ("Lead time (months in advance) is used by the model",
         importances.get("days_before", 0) > 0.02, f"importance = {importances.get('days_before', 0):.3f}"),
        ("Day-level calendar effects are used (dom / dow / holidays)",
         sum(importances.get(c, 0) for c in ("dom", "dow", "is_weekend", "is_holiday", "holiday_dist")) > 0.02,
         f"importance = {sum(importances.get(c, 0) for c in ('dom', 'dow', 'is_weekend', 'is_holiday', 'holiday_dist')):.3f}"),
        ("Stable across CV folds (std/mean of MAE < 25 %)",
         np.std(cv[best]) / np.mean(cv[best]) < 0.25,
         f"{np.std(cv[best]) / np.mean(cv[best]) * 100:.1f} %"),
    ]
    progress("done")
    return ModelResults(models, cv, test_m, test_p, coefs, rf_grid, importances, best, checks)


# --------------------------------------------------------------------------- #
# Step 4 - export of NN-ready data (for training a network in another project)
# --------------------------------------------------------------------------- #
NN_NOMINAL = NOMINAL + ["dow"]
NN_SCALED = ["days_before", "dom", "holiday_dist", "dep_minutes", "duration_min", "dep_period_ord"]
NN_BINARY = ["is_weekend", "is_holiday"]
NN_META = ["scrape_date", "flight_date", "route", "price"]


@dataclass
class NNExport:
    encoded: pd.DataFrame     # meta_* + split/fold + standardised & one-hot features + log_price
    clean: pd.DataFrame       # same rows, unencoded features (for custom encoding / embeddings)
    meta: dict                # feature list, scaler and encoder parameters, split description
    feature_cols: list


def _split_cols(n: int, split: SplitResult) -> tuple[np.ndarray, np.ndarray]:
    """'train'/'test' label and Group K-Fold validation fold (0..k-1, -1 = test) per row."""
    split_lbl = np.full(n, "train", dtype=object)
    split_lbl[split.test_idx] = "test"
    fold = np.full(n, -1, dtype=int)
    for k, (_, val) in enumerate(split.cv_schemes["Group K-Fold (by flight day)"]):
        fold[split.train_idx[val]] = k
    return split_lbl, fold


def nn_export(df: pd.DataFrame, split: SplitResult) -> NNExport:
    """Data ready for a neural network: one-hot nominal features, standardised numeric
    features (scaler fitted on the training rows only, so the test set does not leak),
    the same train/test split and CV folds as steps 2-4, and target log_price."""
    prep = ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), NN_NOMINAL),
        ("num", StandardScaler(), NN_SCALED),
        ("bin", "passthrough", NN_BINARY),
    ], verbose_feature_names_out=False)
    prep.fit(df.iloc[split.train_idx])
    feats = list(prep.get_feature_names_out())
    X = pd.DataFrame(prep.transform(df), columns=feats, index=df.index).astype("float32")

    split_lbl, fold = _split_cols(len(df), split)
    ids = pd.DataFrame({
        "meta_scrape_date": df["scrape_date"].dt.strftime("%Y-%m-%d"),
        "meta_flight_date": df["flight_date"].dt.strftime("%Y-%m-%d"),
        "meta_route": df["route"],
        "meta_price": df["price"],
        "split": split_lbl,
        "fold": fold,
    }, index=df.index)
    encoded = pd.concat([ids, X, df[["log_price"]]], axis=1)

    clean = df[["scrape_date", "flight_date", "route"] + NOMINAL + NUMERIC + ["price", "log_price"]].copy()
    clean["scrape_date"] = clean["scrape_date"].dt.strftime("%Y-%m-%d")
    clean["flight_date"] = clean["flight_date"].dt.strftime("%Y-%m-%d")
    clean.insert(0, "fold", fold)
    clean.insert(0, "split", split_lbl)

    enc, sc = prep.named_transformers_["nom"], prep.named_transformers_["num"]
    meta = {
        "target": "log_price",
        "target_inverse": "price = expm1(log_price)",
        "feature_columns": feats,
        "ignore_columns_for_training": [c for c in ids.columns],
        "one_hot": {c: [str(v) for v in cats] for c, cats in zip(NN_NOMINAL, enc.categories_)},
        "standard_scaler": {c: {"mean": float(m), "std": float(s)}
                            for c, m, s in zip(NN_SCALED, sc.mean_, sc.scale_)},
        "passthrough_binary": NN_BINARY,
        "split": {"method": "GroupShuffleSplit by flight day + route, 20 % test",
                  "cv": "GroupKFold (by flight day), column 'fold' = validation fold, -1 = test",
                  "random_state": RANDOM_STATE,
                  "n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx))},
        "note": "Scaler and encoder were fitted on the training rows only.",
    }
    return NNExport(encoded, clean, meta, feats)


# --------------------------------------------------------------------------- #
# Step 6 - calendar prediction
# --------------------------------------------------------------------------- #
def flight_profiles(df: pd.DataFrame, dest: str, top: int = 8) -> pd.DataFrame:
    """Most frequent cheapest-flight profiles (airline, time, duration) on a route
    with their relative frequency - used to marginalise over unknown future
    airline / departure time."""
    sub = df[df["dest"] == dest]
    prof = (sub.groupby(["airline", "dep_minutes", "duration_min", "dep_period", "dep_period_ord"])
            .size().sort_values(ascending=False).head(top).rename("w").reset_index())
    prof["w"] = prof["w"] / prof["w"].sum()
    return prof


def predict_calendar(df: pd.DataFrame, model, dest: str, year: int, month: int,
                     months_ahead: int, yoy_growth: float, base_year: int,
                     far_mode: str = "flat") -> pd.DataFrame:
    """Predicted price for every day of `month`/`year` on `dest`, when the ticket is
    bought `months_ahead` months before departure."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
    purchase = days - pd.DateOffset(months=months_ahead)
    lead = (days - purchase).days.to_numpy()
    max_lead = int(df["days_before"].max())

    base = pd.DataFrame({"flight_date": days, "dest": dest, "lead_true": lead})
    base = calendar_features(base)
    base["days_before"] = np.minimum(lead, max_lead)
    prof = flight_profiles(df, dest)
    grid = base.merge(prof, how="cross")
    grid = add_model_cols(grid)
    grid["pred"] = np.expm1(model.predict(grid))
    out = (grid.assign(wp=grid["pred"] * grid["w"]).groupby("flight_date")["wp"].sum()
           .rename("price").reset_index())
    out = out.merge(base[["flight_date", "dom", "dow", "lead_true", "is_holiday"]], on="flight_date")

    # Beyond observed lead times trees cannot extrapolate: either keep flat or
    # continue the log-linear trend of the far end of the observed booking curve.
    out["extrapolated"] = out["lead_true"] > max_lead
    if far_mode == "trend" and out["extrapolated"].any():
        slope = booking_curve_slope(df, model, dest, max_lead)
        extra = (out["lead_true"] - max_lead).clip(lower=0)
        out["price"] *= np.exp(np.clip(slope * extra, -0.3, 0.3))

    out["price"] *= (1 + yoy_growth) ** (year - base_year)
    out["purchase_date"] = (out["flight_date"] - pd.DateOffset(months=months_ahead))
    return out


def booking_curve_slope(df, model, dest, max_lead, window=45) -> float:
    """d log(price) / d lead-day over the last `window` observed lead days."""
    leads = np.arange(max(0, max_lead - window), max_lead + 1)
    ref = df[df["dest"] == dest]["flight_date"].iloc[0]
    rows = pd.DataFrame({"flight_date": ref, "dest": dest, "days_before": leads})
    rows = calendar_features(rows).merge(flight_profiles(df, dest), how="cross")
    rows = add_model_cols(rows)
    rows["p"] = np.expm1(model.predict(rows)) * rows["w"]
    curve = rows.groupby("days_before")["p"].sum()
    return float(np.polyfit(curve.index, np.log(curve.to_numpy()), 1)[0])


def booking_curve(df, model, dest, year, month, yoy_growth, base_year, far_mode) -> pd.DataFrame:
    """Average predicted monthly price for each months-in-advance 0..11."""
    rows = []
    for k in range(12):
        c = predict_calendar(df, model, dest, year, month, k, yoy_growth, base_year, far_mode)
        rows.append({"months_ahead": k, "avg": c["price"].mean(), "min": c["price"].min(),
                     "max": c["price"].max(), "extrapolated": bool(c["extrapolated"].any())})
    return pd.DataFrame(rows)
