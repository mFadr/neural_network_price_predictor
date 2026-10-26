"""Flight price prediction app - run `python app.py` and open http://127.0.0.1:8050"""
from __future__ import annotations

import base64
import calendar as pycal
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

import ml_pipeline as mp

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)
DEFAULT_DATASET = "PRG_rep.csv"

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]
TEMPLATE = "plotly_white"

# In-memory cache: dataset name -> results dict.
CACHE: dict[str, dict] = {}
LOCK = threading.Lock()


def list_datasets() -> list[str]:
    files = sorted(p.name for p in ROOT.glob("*.csv")) + sorted(f"uploads/{p.name}" for p in UPLOADS.glob("*.csv"))
    return files


def run_pipeline(name: str) -> dict:
    with LOCK:
        if name in CACHE:
            return CACHE[name]
        raw = mp.load_raw(str(ROOT / name))
        df, rep = mp.prepare_data(raw)
        split = mp.split_and_cv(df)
        res = mp.train_models(df, split)
        CACHE[name] = {"raw": raw, "df": df, "rep": rep, "split": split, "res": res, "cal": {}}
        return CACHE[name]


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def card(title, value, sub=""):
    return html.Div(className="card", children=[
        html.Div(title, className="card-title"), html.Div(value, className="card-value"),
        html.Div(sub, className="card-sub")])


def table(df: pd.DataFrame, fmt: dict | None = None):
    fmt = fmt or {}
    head = html.Tr([html.Th(c) for c in df.columns])
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:,.2f}"
            elif isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%d")
            cells.append(html.Td(str(v)))
        rows.append(html.Tr(cells))
    return html.Div(className="table-wrap", children=html.Table([html.Thead(head), html.Tbody(rows)]))


def section(title, *children, note=None):
    kids = [html.H3(title)]
    if note:
        kids.append(html.P(note, className="note"))
    return html.Div(className="section", children=kids + list(children))


def graph(fig, height=380):
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=50, r=20, t=50, b=40),
                      colorway=PALETTE, font=dict(family="Inter, Segoe UI, sans-serif", size=12))
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def grid2(*children):
    return html.Div(className="grid2", children=list(children))


def nn_data(c) -> mp.NNExport:
    """NN-ready export for a cached dataset (built once, on first use)."""
    if "nn" not in c:
        c["nn"] = mp.nn_export(c["df"], c["split"])
    return c["nn"]


# --------------------------------------------------------------------------- #
# Tab content builders
# --------------------------------------------------------------------------- #
def tab_prep(c):
    df, rep, raw = c["df"], c["rep"], c["raw"]
    fixes = pd.DataFrame(list(rep.fixes.items()), columns=["Cleaning step", "Rows affected"])
    clean_cols = ["scrape_date", "flight_date", "route", "airline", "dep_minutes", "dep_period",
                  "duration_min", "days_before", "dom", "dow", "is_weekend", "holiday_dist", "price"]

    fig_box = px.box(df, x="dest", y="price", color="dest", points=False,
                     title="Price distribution per route (after cleaning)")
    fig_box.update_layout(showlegend=False, xaxis_title="Destination", yaxis_title="Price (USD)")

    wk = df.assign(lead_week=df["days_before"] // 7).groupby(["dest", "lead_week"])["price"].mean().reset_index()
    fig_heat = go.Figure(go.Heatmap(x=wk["lead_week"], y=wk["dest"], z=wk["price"], colorscale="YlOrRd",
                                    colorbar=dict(title="USD")))
    fig_heat.update_layout(title="Mean price by weeks before departure", xaxis_title="Weeks before departure",
                           yaxis_title="Destination")

    fig_hist = make_subplots(1, 2, subplot_titles=("price (raw scale)", "log(1+price) - model target"))
    fig_hist.add_trace(go.Histogram(x=df["price"], nbinsx=60, marker_color=PALETTE[0]), 1, 1)
    fig_hist.add_trace(go.Histogram(x=df["log_price"], nbinsx=60, marker_color=PALETTE[2]), 1, 2)
    fig_hist.update_layout(showlegend=False, title="Target transformation")

    dow = df.groupby("dow")["price"].mean()
    fig_dow = go.Figure(go.Bar(x=[pycal.day_abbr[i] for i in dow.index], y=dow.values, marker_color=PALETTE[0]))
    fig_dow.update_layout(title="Mean price by day of week of departure (ordinal feature)", yaxis_title="USD")

    return html.Div([
        html.Div(className="cards", children=[
            card("Raw rows", f"{rep.n_raw:,}"), card("Clean rows", f"{rep.n_clean:,}",
                                                      f"{rep.n_clean / rep.n_raw:.0%} kept"),
            card("Routes", f"{df['route'].nunique()}", ", ".join(sorted(df["dest"].unique()))),
            card("Flight dates", f"{df['flight_date'].min():%Y-%m-%d}", f"to {df['flight_date'].max():%Y-%m-%d}"),
            card("Lead time observed", f"0-{df['days_before'].max()} days",
                 f"≈ {df['days_before'].max() / 30.4:.1f} months"),
        ]),
        grid2(section("Column scales & unification", table(mp.column_types_table()),
                      note="Nominal columns are one-hot encoded; ordinal ones keep their natural order as integers; "
                           "ratio columns are parsed to numbers."),
              section("Cleaning log", table(fixes))),
        section("Raw input (first rows)", table(c["raw"].head(6))),
        section("Unified data (first rows)", table(df[clean_cols].head(6))),
        grid2(graph(fig_box), graph(fig_heat)),
        grid2(graph(fig_hist), graph(fig_dow)),
    ])


def tab_cv(c):
    df, split = c["df"], c["split"]
    train, test = df.iloc[split.train_idx], df.iloc[split.test_idx]

    # Hold-out split visual: which (flight day, route) groups went to test.
    m = df.assign(set=np.where(np.isin(np.arange(len(df)), split.test_idx), 1, 0))
    g = m.groupby(["dest", "dom"])["set"].max().unstack()
    fig_split = go.Figure(go.Heatmap(z=g.values, x=g.columns, y=g.index, zmin=0, zmax=1, showscale=False,
                                     colorscale=[[0, "#bfdbfe"], [0.5, "#bfdbfe"], [0.5, "#f97316"], [1, "#f97316"]],
                                     hovertemplate="%{y} day %{x}: %{z}<extra>1 = test</extra>"))
    fig_split.update_layout(title=f"Hold-out split by flight day × route: train {len(train):,} rows (blue) / "
                                  f"test {len(test):,} rows (orange)", xaxis_title="Day of month", yaxis_title="Route")

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=train["price"], name="train", histnorm="probability density",
                                    opacity=0.6, nbinsx=60))
    fig_dist.add_trace(go.Histogram(x=test["price"], name="test", histnorm="probability density",
                                    opacity=0.6, nbinsx=60))
    fig_dist.update_layout(barmode="overlay", title="Train vs test price distribution", xaxis_title="USD")

    # Fold structure for each CV scheme.
    names = list(split.cv_schemes)
    fig_folds = make_subplots(len(names), 1, subplot_titles=names, vertical_spacing=0.12)
    tr = train.reset_index(drop=True)
    for i, name in enumerate(names, 1):
        folds = split.cv_schemes[name]
        if name.startswith("K-Fold"):
            n = 150
            z = np.zeros((len(folds), n))
            for f, (_, b) in enumerate(folds):
                z[f, np.intersect1d(b, np.arange(n))] = 1
            x, xt = np.arange(n), "first 150 training rows"
        elif name.startswith("Group"):
            keys = (tr["dest"] + " " + tr["dom"].astype(str).str.zfill(2))
            uniq = np.sort(keys.unique())
            z = np.zeros((len(folds), len(uniq)))
            pos = {k: j for j, k in enumerate(uniq)}
            for f, (_, b) in enumerate(folds):
                for k in keys.iloc[b].unique():
                    z[f, pos[k]] = 1
            x, xt = uniq, "route + flight day group"
        else:
            dates = np.sort(tr["scrape_date"].unique())
            pos = {d: j for j, d in enumerate(dates)}
            z = np.full((len(folds), len(dates)), np.nan)
            for f, (a, b) in enumerate(folds):
                for d in tr["scrape_date"].iloc[a].unique():
                    z[f, pos[d]] = 0
                for d in tr["scrape_date"].iloc[b].unique():
                    z[f, pos[d]] = 1
            x, xt = pd.to_datetime(dates), "scrape date"
        fig_folds.add_trace(go.Heatmap(z=z, x=x, y=[f"fold {k + 1}" for k in range(len(folds))], zmin=0, zmax=1,
                                       showscale=False, colorscale=[[0, "#bfdbfe"], [1, "#f97316"]],
                                       hovertemplate="%{y} %{x}: %{z}<extra>1 = validation</extra>"), i, 1)
        fig_folds.update_xaxes(title_text=xt, showticklabels=not name.startswith("Group"), row=i, col=1)

    fig_folds.update_layout(title="Cross-validation fold structure (blue = train, orange = validation)")

    fig_scores = go.Figure()
    for name, sc in split.cv_scores.items():
        fig_scores.add_trace(go.Bar(name=name, x=[f"fold {i + 1}" for i in range(len(sc))], y=sc,
                                    text=[f"{v:.1f}" for v in sc], textposition="outside"))
    fig_scores.update_layout(barmode="group", title="Validation MAE per fold (Ridge baseline) - USD",
                             yaxis_title="MAE (USD)", legend=dict(orientation="h", y=-0.2))

    summ = pd.DataFrame([{"Scheme": k, "Mean MAE": np.mean(v), "Std MAE": np.std(v)}
                         for k, v in split.cv_scores.items()])
    return html.Div([
        section("Why these schemes?", html.Ul([
            html.Li("Hold-out test set: 20 % of (route, flight day) groups are locked away and only used for the "
                    "final evaluation in steps 3-4."),
            html.Li("K-Fold on random rows looks good but leaks: the same flight is scraped every day with nearly "
                    "identical prices, so its neighbours sit in the training folds."),
            html.Li("Group K-Fold keeps every observation of a (route, flight day) in one fold - this matches the "
                    "real task (predict days never seen) and is used for model selection."),
            html.Li("Time-Series split trains on earlier scrape dates only; folds differ strongly because the lead "
                    "time distribution shifts over the scrape window."),
        ])),
        grid2(graph(fig_split, 330), graph(fig_dist, 330)),
        graph(fig_folds, 620),
        grid2(graph(fig_scores, 400), section("CV summary (Ridge baseline)", table(summ))),
    ])


def tab_regression(c):
    df, split, res = c["df"], c["split"], c["res"]
    test = df.iloc[split.test_idx]
    pred = res.test_pred["Ridge regression"]
    met = res.test["Ridge regression"]

    lim = [0, float(max(test["price"].max(), pred.max()))]
    fig_sc = go.Figure(go.Scattergl(x=test["price"], y=pred, mode="markers",
                                    marker=dict(size=4, opacity=0.35, color=PALETTE[0])))
    fig_sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#64748b")))
    fig_sc.update_layout(title="Actual vs predicted (hold-out test)", xaxis_title="Actual USD",
                         yaxis_title="Predicted USD", showlegend=False)

    fig_res = go.Figure(go.Histogram(x=test["price"] - pred, nbinsx=70, marker_color=PALETTE[3]))
    fig_res.update_layout(title="Residuals (actual - predicted)", xaxis_title="USD")

    co = res.ridge_coefs
    co = pd.concat([co.head(10), co.tail(10)])
    fig_co = go.Figure(go.Bar(x=co.values, y=[n.split("__", 1)[1] for n in co.index], orientation="h",
                              marker_color=np.where(co.values > 0, PALETTE[3], PALETTE[2])))
    fig_co.update_layout(title="Largest Ridge coefficients (effect on log price)")

    curve = test.assign(pred=pred).groupby("days_before")[["price", "pred"]].mean().reset_index()
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve["days_before"], y=curve["price"], name="actual mean",
                                   mode="markers", marker=dict(size=5)))
    fig_curve.add_trace(go.Scatter(x=curve["days_before"], y=curve["pred"], name="Ridge mean", mode="lines"))
    fig_curve.update_layout(title="Booking curve: price vs days before departure", xaxis_title="days before",
                            yaxis_title="USD", xaxis_autorange="reversed")

    return html.Div([
        section("Linear regression (Ridge, α=1) on log(1+price)",
                html.P("Features: one-hot route / airline / part-of-day / weekday, standardised numeric features "
                       "and a quadratic lead-time term. It is the interpretable baseline the ML models must beat."),
                html.Div(className="cards", children=[
                    card("Test MAE", f"${met['MAE']:.2f}"), card("Test MAPE", f"{met['MAPE']:.1f} %"),
                    card("Test R²", f"{met['R2']:.3f}"),
                    card("CV MAE (Group K-Fold)", f"${np.mean(res.cv['Ridge regression']):.2f}",
                         f"± {np.std(res.cv['Ridge regression']):.2f}")])),
        grid2(graph(fig_sc), graph(fig_res)),
        grid2(graph(fig_co, 480), graph(fig_curve, 480)),
    ])


def tab_ml(c):
    df, split, res = c["df"], c["split"], c["res"]
    test = df.iloc[split.test_idx]

    rows = []
    for name, m in res.test.items():
        cv = res.cv.get(name)
        rows.append({"Model": name + (" ★" if name == res.best else ""),
                     "CV MAE": f"{np.mean(cv):.2f} ± {np.std(cv):.2f}" if cv else "-",
                     "Test MAE": m["MAE"], "Test MAPE %": m["MAPE"], "Test R²": m["R2"]})
    comp = pd.DataFrame(rows)

    fig_cv = go.Figure()
    for name, sc in res.cv.items():
        fig_cv.add_trace(go.Box(y=sc, name=name, boxpoints="all", jitter=0.3))
    fig_cv.update_layout(title="Cross-validated MAE per model (Group K-Fold, 5 folds)", yaxis_title="USD",
                         showlegend=False)

    g = res.rf_grid.pivot(index="max_depth", columns="min_samples_leaf", values="CV MAE")
    fig_grid = go.Figure(go.Heatmap(z=g.values, x=[str(v) for v in g.columns], y=g.index, colorscale="Blues_r",
                                    text=np.round(g.values, 2), texttemplate="%{text}", colorbar=dict(title="MAE")))
    fig_grid.update_layout(title="Random Forest grid search (3-fold CV MAE)", xaxis_title="min_samples_leaf",
                           yaxis_title="max_depth")

    imp = res.importances
    fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PALETTE[0]))
    fig_imp.update_layout(title="Random Forest feature importance (one-hot columns aggregated)")

    best_pred = res.test_pred[res.best]
    lim = [0, float(max(test["price"].max(), best_pred.max()))]
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scattergl(x=test["price"], y=res.test_pred["Ridge regression"], mode="markers",
                                  name="Ridge", marker=dict(size=3, opacity=0.25, color="#94a3b8")))
    fig_sc.add_trace(go.Scattergl(x=test["price"], y=best_pred, mode="markers", name=res.best,
                                  marker=dict(size=4, opacity=0.4, color=PALETTE[0])))
    fig_sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#64748b"),
                                showlegend=False))
    fig_sc.update_layout(title=f"Actual vs predicted: {res.best} vs Ridge", xaxis_title="Actual USD",
                         yaxis_title="Predicted USD")

    lw = (test["days_before"] // 14) * 2
    fig_err = go.Figure()
    for name in ["Ridge regression", "Random Forest", "Gradient Boosting"]:
        e = pd.Series(np.abs(test["price"].to_numpy() - res.test_pred[name])).groupby(lw.to_numpy()).mean()
        fig_err.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines+markers", name=name))
    fig_err.update_layout(title="Test MAE by weeks before departure", xaxis_title="weeks before departure",
                          yaxis_title="MAE (USD)")

    checks = html.Ul(className="checks", children=[
        html.Li([html.Span("✔" if ok else "✘", className="ok" if ok else "bad"), f" {req} — ", html.B(detail)])
        for req, ok, detail in res.checks])

    return html.Div([
        grid2(section("Model comparison", table(comp, {"Test MAE": lambda v: f"{v:.2f}",
                                                       "Test MAPE %": lambda v: f"{v:.1f}",
                                                       "Test R²": lambda v: f"{v:.3f}"}),
                      note=f"★ selected model for the price calendar: {res.best} (lowest cross-validated MAE)."),
              section("Requirement checks", checks)),
        grid2(graph(fig_cv), graph(fig_grid)),
        grid2(graph(fig_sc), graph(fig_imp)),
        graph(fig_err),
        nn_export_section(c),
    ])


def nn_export_section(c):
    nn = nn_data(c)
    enc = nn.encoded
    n_tr, n_te = int((enc["split"] == "train").sum()), int((enc["split"] == "test").sum())
    preview = enc.head(5)[["split", "fold", "meta_route", "meta_price"] + nn.feature_cols[:6] + ["log_price"]]
    return section(
        "Export data for a neural network",
        html.Div(className="cards", children=[
            card("Rows", f"{len(enc):,}", f"{n_tr:,} train · {n_te:,} test"),
            card("Input features", str(len(nn.feature_cols)), "one-hot + standardised"),
            card("Target", "log_price", "price = expm1(log_price)"),
        ]),
        html.Div(className="export-buttons", children=[
            html.Button("NN-ready CSV (encoded + scaled)", id="nn-dl-encoded", className="btn"),
            html.Button("Clean features CSV (not encoded)", id="nn-dl-clean", className="btn secondary"),
            html.Button("Preprocessing metadata (JSON)", id="nn-dl-meta", className="btn secondary"),
        ]),
        table(preview, {"fold": str, "meta_price": lambda v: f"{v:.2f}"}),
        note=("Same rows, train/test split and Group K-Fold folds as steps 2-4, so a PyTorch model trained "
              "in another project can be compared fairly with the models above. Columns 'split' (train/test) "
              "and 'fold' (CV validation fold, -1 = test) define the split; 'meta_*' columns are for "
              "reference only, not model inputs. Scaler/encoder are fitted on training rows only - the JSON "
              "holds their parameters, so new data (e.g. calendar days) can be encoded the same way."),
    )


def tab_calendar(c):
    df = c["df"]
    base_year = int(df["flight_date"].dt.year.min())
    months = sorted(df["flight_date"].dt.month.unique())
    routes = sorted(df["dest"].unique())
    origin = df["origin"].iloc[0]
    return html.Div([
        html.Div(className="controls", children=[
            html.Div([html.Label("Flight route"),
                      dcc.Dropdown(id="cal-route", options=[{"label": f"{origin} → {d}", "value": d} for d in routes],
                                   value=routes[0], clearable=False)]),
            html.Div([html.Label("Travel month"),
                      dcc.Dropdown(id="cal-month", options=[{"label": pycal.month_name[m], "value": m}
                                                            for m in months], value=months[0], clearable=False)]),
            html.Div([html.Label("Model"),
                      dcc.Dropdown(id="cal-model", options=list(c["res"].models), value=c["res"].best,
                                   clearable=False)]),
        ]),
        html.Div(className="controls", children=[
            html.Div(className="wide", children=[
                html.Label("Travel year"),
                dcc.Slider(id="cal-year", min=base_year, max=base_year + 2, step=1, value=base_year,
                           marks={y: str(y) for y in range(base_year, base_year + 3)})]),
            html.Div(className="wide", children=[
                html.Label("Purchase: months in advance"),
                # Reversed: left = 11 months ahead, right = 0 (buying just before departure).
                dcc.Slider(id="cal-ahead", min=0, max=11, step=1, value=3, reverse=True,
                           marks={k: str(k) for k in range(12)})]),
        ]),
        html.Div(className="controls", children=[
            html.Div([html.Label("Year-over-year price growth"),
                      dcc.Slider(id="cal-growth", min=0, max=10, step=0.5, value=3,
                                 marks={v: f"{v}%" for v in range(0, 11, 2)})]),
            html.Div([html.Label("Lead times beyond observed data"),
                      dcc.RadioItems(id="cal-far", inline=True, value="flat",
                                     options=[{"label": " flat (hold last observed level)", "value": "flat"},
                                              {"label": " continue booking-curve trend", "value": "trend"}])]),
        ]),
        dcc.Loading(html.Div(id="cal-output")),
    ])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = Dash(__name__, title="Flight Price Predictor", suppress_callback_exceptions=True)

app.layout = html.Div(className="page", children=[
    html.Header(children=[
        html.H1("Flight Price Predictor"),
        html.P("Machine-learning forecast of the daily price level per route, by travel year and "
               "how many months in advance the ticket is bought."),
    ]),
    html.Div(className="controls top", children=[
        html.Div([html.Label("Dataset"),
                  dcc.Dropdown(id="dataset", options=list_datasets(), value=DEFAULT_DATASET, clearable=False)]),
        html.Div([html.Label("…or upload a CSV (same 8-column format)"),
                  dcc.Upload(id="upload", className="upload",
                             children=html.Div(["Drag & drop or ", html.A("select file")]))]),
        html.Div(id="status", className="status"),
    ]),
    dcc.Tabs(id="tabs", value="cal", className="tabs", children=[
        dcc.Tab(label="1 · Data preparation", value="prep"),
        dcc.Tab(label="2 · Train/test & cross-validation", value="cv"),
        dcc.Tab(label="3 · Regression", value="reg"),
        dcc.Tab(label="4 · ML models (RF, GB)", value="ml"),
        dcc.Tab(label="6 · Price calendar", value="cal"),
    ]),
    dcc.Loading(html.Div(id="tab-content", className="content"), type="circle"),
    dcc.Download(id="nn-download"),
])


@app.callback(Output("dataset", "options"), Output("dataset", "value"),
              Input("upload", "contents"), State("upload", "filename"), prevent_initial_call=True)
def on_upload(contents, filename):
    if not contents:
        return no_update, no_update
    data = base64.b64decode(contents.split(",", 1)[1])
    name = Path(filename).stem + ".csv"
    (UPLOADS / name).write_bytes(data)
    return list_datasets(), f"uploads/{name}"


@app.callback(Output("tab-content", "children"), Output("status", "children"),
              Input("tabs", "value"), Input("dataset", "value"))
def render_tab(tab, dataset):
    try:
        c = run_pipeline(dataset)
    except Exception as e:  # surface data errors in the UI instead of a blank page
        return html.Div(f"Could not process {dataset}: {e}", className="error"), "error"
    res = c["res"]
    status = f"{dataset} · best model: {res.best} · test MAPE {res.test[res.best]['MAPE']:.1f} %"
    builders = {"prep": tab_prep, "cv": tab_cv, "reg": tab_regression, "ml": tab_ml, "cal": tab_calendar}
    return builders[tab](c), status


@app.callback(Output("nn-download", "data"),
              Input("nn-dl-encoded", "n_clicks"), Input("nn-dl-clean", "n_clicks"), Input("nn-dl-meta", "n_clicks"),
              State("dataset", "value"), prevent_initial_call=True)
def download_nn(_enc, _clean, _meta, dataset):
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update
    nn = nn_data(run_pipeline(dataset))
    stem = Path(dataset).stem
    if ctx.triggered_id == "nn-dl-encoded":
        return dcc.send_data_frame(nn.encoded.to_csv, f"{stem}_nn_ready.csv", index=False)
    if ctx.triggered_id == "nn-dl-clean":
        return dcc.send_data_frame(nn.clean.to_csv, f"{stem}_nn_clean.csv", index=False)
    return dict(content=json.dumps(nn.meta, indent=2), filename=f"{stem}_nn_meta.json")


def calendars_for(c, dest, year, month, model_name, growth, far):
    """All 12 months-ahead calendars for one route/year (cached)."""
    key = (dest, year, month, model_name, growth, far)
    if key not in c["cal"]:
        df = c["df"]
        base_year = int(df["flight_date"].dt.year.min())
        model = c["res"].models[model_name]
        c["cal"][key] = {k: mp.predict_calendar(df, model, dest, year, month, k, growth / 100, base_year, far)
                         for k in range(12)}
    return c["cal"][key]


def level_of(price, lo, hi):
    t = (price - lo) / max(hi - lo, 1e-9)
    return "Low" if t < 1 / 3 else ("Medium" if t < 2 / 3 else "High")


@app.callback(Output("cal-output", "children"),
              Input("cal-route", "value"), Input("cal-month", "value"), Input("cal-year", "value"),
              Input("cal-ahead", "value"), Input("cal-model", "value"), Input("cal-growth", "value"),
              Input("cal-far", "value"), State("dataset", "value"))
def render_calendar(dest, month, year, k, model_name, growth, far, dataset):
    c = run_pipeline(dataset)
    cals = calendars_for(c, dest, year, month, model_name, growth, far)
    cal = cals[k]
    allp = np.concatenate([v["price"].to_numpy() for v in cals.values()])
    lo, hi = float(allp.min()), float(allp.max())   # fixed colour scale across the slider

    # Calendar grid: rows = weeks, columns = Mon..Sun.
    first_dow = pycal.monthrange(year, month)[0]
    n_weeks = (first_dow + len(cal) + 6) // 7
    z = np.full((n_weeks, 7), np.nan)
    text = np.full((n_weeks, 7), "", dtype=object)
    hover = np.full((n_weeks, 7), "", dtype=object)
    for _, r in cal.iterrows():
        i = first_dow + int(r["dom"]) - 1
        w, d = divmod(i, 7)
        z[w, d] = r["price"]
        star = "*" if r["extrapolated"] else ""
        text[w, d] = f"<b>{int(r['dom'])}</b><br>${r['price']:.0f}{star}<br>{level_of(r['price'], lo, hi)}"
        hover[w, d] = (f"{r['flight_date']:%a %d %b %Y}<br>buy on {r['purchase_date']:%d %b %Y} "
                       f"({int(r['lead_true'])} days ahead)<br>predicted ${r['price']:.2f}"
                       + ("<br>lead time beyond observed data" if r["extrapolated"] else ""))
    fig = go.Figure(go.Heatmap(z=z, x=list(pycal.day_abbr), y=[f"W{i + 1}" for i in range(n_weeks)],
                               text=text, texttemplate="%{text}", customdata=hover,
                               hovertemplate="%{customdata}<extra></extra>", zmin=lo, zmax=hi,
                               colorscale=[[0, "#16a34a"], [0.5, "#facc15"], [1, "#dc2626"]], xgap=4, ygap=4,
                               colorbar=dict(title="USD")))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(side="top")
    fig.update_layout(template=TEMPLATE, height=520, margin=dict(l=20, r=20, t=70, b=20),
                      title=f"{c['df']['origin'].iloc[0]} → {dest} · {pycal.month_name[month]} {year} · bought {k} month"
                            f"{'s' if k != 1 else ''} in advance", plot_bgcolor="white")

    # Booking curve: average monthly price for each months-ahead value.
    bc = pd.DataFrame([{"k": kk, "avg": v["price"].mean(), "min": v["price"].min(), "max": v["price"].max(),
                        "ext": bool(v["extrapolated"].any())} for kk, v in cals.items()])
    fb = go.Figure()
    fb.add_trace(go.Scatter(x=bc["k"], y=bc["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fb.add_trace(go.Scatter(x=bc["k"], y=bc["min"], fill="tonexty", fillcolor="rgba(37,99,235,0.12)",
                            line=dict(width=0), name="min-max day"))
    fb.add_trace(go.Scatter(x=bc["k"], y=bc["avg"], mode="lines+markers", name="month average",
                            line=dict(color=PALETTE[0]),
                            marker=dict(symbol=np.where(bc["ext"], "circle-open", "circle"), size=8)))
    fb.add_vline(x=k, line_dash="dot", line_color="#64748b")
    fb.update_layout(template=TEMPLATE, height=360, title="Price by purchase timing (open markers = extrapolated)",
                     xaxis=dict(title="months in advance", autorange="reversed", dtick=1), yaxis_title="USD",
                     margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.25))

    cheapest = cal.loc[cal["price"].idxmin()]
    priciest = cal.loc[cal["price"].idxmax()]
    best_k = int(bc.loc[bc["avg"].idxmin(), "k"])
    max_lead = int(c["df"]["days_before"].max())
    notes = []
    if cal["extrapolated"].any():
        notes.append(f"* The data only contains purchases up to {max_lead} days (~{max_lead / 30.4:.1f} months) "
                     f"before departure. Longer lead times are extrapolated ({far}).")
    base_year = int(c["df"]["flight_date"].dt.year.min())
    if year > base_year:
        notes.append(f"The dataset contains a single year ({base_year}); {year} uses the {base_year} model with the "
                     f"weekday/holiday calendar of {year} and {growth}% yearly price growth.")
    return html.Div([
        html.Div(className="cards", children=[
            card("Month average", f"${cal['price'].mean():.0f}", f"bought {k} months ahead"),
            card("Cheapest day", f"{cheapest['flight_date']:%a %d %b}", f"${cheapest['price']:.0f}"),
            card("Most expensive day", f"{priciest['flight_date']:%a %d %b}", f"${priciest['price']:.0f}"),
            card("Best time to buy", f"{best_k} months ahead", f"avg ${bc['avg'].min():.0f}"),
        ]),
        grid2(dcc.Graph(figure=fig, config={"displaylogo": False}),
              dcc.Graph(figure=fb, config={"displaylogo": False})),
        html.Div([html.P(n, className="note") for n in notes]),
    ])


if __name__ == "__main__":
    print("Training models on", DEFAULT_DATASET, "...")
    run_pipeline(DEFAULT_DATASET)
    print("Open http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
