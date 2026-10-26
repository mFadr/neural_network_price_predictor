# ✈️ Neural Network Flight Price Predictor

An interactive machine-learning web app (built with **Dash + scikit-learn + Plotly**) that predicts flight ticket prices
from scraped price data. You pick a route, a travel month and year, and how many months in advance you plan to buy the
ticket. The app then shows a **price calendar** with the predicted price for every day of that month, plus the
cheapest time to buy.

The app also walks through the whole ML workflow in separate tabs: data cleaning, train/test split and
cross-validation, a linear regression baseline, tree-based ensemble models, and an export of NN-ready data.

---

## 🚀 Quick start (how to run it)

### 1. Requirements

- **Python 3.10+** (developed on Python 3.13)
- `pip`

### 2. Install the dependencies

Open a terminal in the project folder:

```bash
cd flight_price_predictor

# (optional but recommended) create a virtual environment
python -m venv .venv

# activate it
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   Windows (cmd):
.venv\Scripts\activate.bat
#   macOS / Linux:
source .venv/bin/activate

# install the packages
pip install -r requirements.txt
```

### 3. Start the app

```bash
python app.py
```

At startup the app trains all models on the default dataset (`PRG_rep.csv`). This takes from a few seconds up to about
a minute, depending on your machine. When it's done you'll see:

```
Training models on PRG_rep.csv ...
Open http://127.0.0.1:8050
```

### 4. Open it in the browser

Go to **http://127.0.0.1:8050**

To stop the server, press `Ctrl + C` in the terminal.

---

## 📁 Project structure

```
flight_price_predictor/
├── app.py              # Dash web application (UI, charts, callbacks)
├── ml_pipeline.py      # Data cleaning, feature engineering, CV, models, calendar prediction
├── requirements.txt    # Python dependencies
├── PRG_rep.csv         # Default dataset - flights from Prague (PRG)
├── uploads/            # Uploaded datasets are saved here (e.g. VIE_rep.csv - flights from Vienna)
└── assets/
    └── style.css       # App styling (Dash loads it automatically)
```

---

## 📊 Data format

Each dataset is a CSV with **8 columns**. A header row is optional and gets detected automatically.

| # | Column        | Example      | Meaning                                         |
|---|---------------|--------------|-------------------------------------------------|
| 1 | `scrape_date` | `2025-09-01` | Date the price was collected (= purchase date)  |
| 2 | `flight_date` | `2026-01-01` | Departure date                                  |
| 3 | `origin`      | `PRG`        | Origin airport (IATA code)                      |
| 4 | `dest`        | `AMS`        | Destination airport (IATA code)                 |
| 5 | `dep_time`    | `09:20 AM`   | Departure time (12h or 24h format)              |
| 6 | `duration`    | `1h 45min`   | Flight duration                                 |
| 7 | `price_raw`   | `US$59.47`   | Ticket price (USD)                              |
| 8 | `airline`     | `Easyjet`    | Airline                                         |

Example rows:

```
2025-09-01,2026-01-01,PRG,AMS,09:20 AM,1h 45min,US$59.47,Easyjet
2025-09-01,2026-01-02,PRG,AMS,09:20 AM,1h 45min,US$71.83,Easyjet
```

The included datasets cover flights in **January 2026** from **Prague (PRG)** and **Vienna (VIE)** to
Amsterdam (AMS), Athens (ATH), Barcelona (BCN), Copenhagen (CPH), Rome (FCO) and London (LON).

### Using your own data

- Pick a dataset from the **Dataset** dropdown at the top of the app, or
- **drag & drop** a CSV into the upload box. The file is saved to `uploads/` and loaded right away.

Every `.csv` in the project root and in `uploads/` shows up in the dropdown. Pipeline results are cached in memory, so
switching back to a dataset you already used is instant. The first load of a new dataset retrains all the models.

---

## 🧠 How it works: the ML pipeline (`ml_pipeline.py`)

### Step 1: Data preparation (`prepare_data`)
Cleans the raw scraped data and unifies its formats:
- splits merged price/airline cells (e.g. `US$36.92Wizz Air`)
- repairs corrupted scrape dates by forward-filling, and drops unparseable flight dates
- turns prices (`US$59.47`, `$59.47`) into numbers
- converts both 12h (`09:20 AM`) and 24h (`17:05`) departure times to minutes after midnight, plus a part of day
  (night < morning < afternoon < evening)
- parses durations (`1h 45min` → 105 minutes)
- groups rare airlines (< 0.5 % of rows) into `Other`
- removes duplicates, rows scraped after departure, and extreme price outliers (log-IQR rule per route, k = 3)

**Engineered features:** `days_before` (lead time), day of month, day of week, weekend flag, holiday flag, distance
to the nearest public holiday (CZ + common European fixed dates), departure minutes, duration, and part of day.
**Target:** `log(1 + price)`.

### Step 2: Train/test split and cross-validation (`split_and_cv`)
- **Hold-out test set:** 20 % of the (route, flight day) groups are never used in training.
- Three CV schemes are compared, using the Ridge baseline:
  - **K-Fold (random rows):** looks good but leaks, because the same flight is scraped every day.
  - **Group K-Fold (by flight day):** used for model selection, since it matches the real task.
  - **Time-Series split (by scrape date):** trains only on earlier observations.

### Step 3: Regression baseline
**Ridge regression** (α = 1) with one-hot categorical features, standardized numeric features and a quadratic
lead-time term. It's the interpretable baseline.

### Step 4: Ensemble ML models (`train_models`)
- **Random Forest:** grid search over `max_depth` × `min_samples_leaf`, then retrained with the best parameters
- **Gradient Boosting** (`HistGradientBoostingRegressor`)
- **Naive baseline:** mean price per route × lead week

The model with the lowest cross-validated MAE is picked as the **best model**. The app also runs automatic
**requirement checks**: the best model must beat the baselines, reach test MAPE < 25 %, use lead time and calendar
features, and be stable across folds.

### NN data export (`nn_export`)
Exports data ready for training a neural network in another project (e.g. PyTorch). It uses the same rows,
train/test split and CV folds as the models above:
- **NN-ready CSV:** one-hot and standardized features (scaler fitted on the training rows only), plus `split`
  and `fold` columns
- **Clean features CSV:** the same features without encoding
- **Metadata JSON:** feature list, encoder categories, and scaler means/stds, so new data can be encoded the same way

### Step 6: Price calendar (`predict_calendar`)
For each day of the chosen month, the model predicts the price when buying *N* months in advance. Because the future
airline and departure time are unknown, the prediction is averaged over the most common flight profiles on the route.
- **Lead times beyond the observed data** are either held *flat* or continue the *booking-curve trend*.
- **Future years** apply a year-over-year price growth (adjustable, 0–10 %) and use that year's weekday/holiday
  calendar.

---

## 🖥️ App tabs

| Tab | What it shows |
|-----|---------------|
| **1 · Data preparation** | Raw vs. clean data, cleaning log, column scales, price distributions, heatmap of price by weeks before departure |
| **2 · Train/test & CV** | Hold-out split visualization, CV fold structure for all three schemes, MAE per fold |
| **3 · Regression** | Ridge metrics (MAE, MAPE, R²), actual vs. predicted, residuals, coefficients, booking curve |
| **4 · ML models (RF, GB)** | Model comparison table, requirement checks, RF grid search, feature importance, error by lead time, **NN data export buttons** |
| **6 · Price calendar** *(default)* | Interactive calendar heatmap (green = cheap, red = expensive), cheapest/most expensive day, best time to buy, price vs. purchase timing |

**Price calendar controls:** route, travel month, model, travel year, months in advance (0–11), year-over-year growth,
and extrapolation mode. Values marked `*` are extrapolated beyond the lead times in the data.

---

## 🛠️ Tech stack

- [Dash](https://dash.plotly.com/): web UI
- [Plotly](https://plotly.com/python/): interactive charts
- [scikit-learn](https://scikit-learn.org/): models, preprocessing, cross-validation
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/): data handling

---

## ⚠️ Notes and limitations

- The datasets cover **one month of flights (January 2026)**, so predictions for other months and years are
  extrapolations based on calendar features and the growth assumption.
- Tree models can't extrapolate past the longest lead time in the data. That's why the *flat / trend* option exists.
- The server runs locally on `127.0.0.1:8050` with `debug=False`. To use a different port, change the last line of
  `app.py`.

*Built by Matěj Fadrhons  [github.com/mFadr](https://github.com/mFadr)*
