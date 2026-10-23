# BP_nn — Flight Price Prediction from Prague

A small PyTorch project that learns flight ticket prices from scraped data for routes departing Prague (PRG). It then uses the trained neural network to find the **cheapest combination of flight date, flight, and booking date**.

The project has three layers, each built on the previous one:

| File | Purpose |
|------|---------|
| `nn.py` | Basic feed-forward neural network: data loading, model definition, training, and a table of the cheapest predicted flight per route. |
| `interactive_interpreter.py` | Interactive terminal menu for exploring the output of `nn.py` (full table, per-route lookup, potential savings). |
| `flight_agent.py` | A "flight agent" that reuses the `nn.py` network as an ensemble, searches **future** dates, and recommends the best flight and the best day to buy it. |
| `PRG_rep_nn_ready.csv` | Preprocessed, model-ready dataset (features already encoded and scaled). |
| `prompt.txt` | The original task description used to create `flight_agent.py`. |

---

## Dataset

`PRG_rep_nn_ready.csv` has **19,209 rows** and 47 columns. Each row is one observed flight offer.

- **Scrape dates:** 2025-09-01 → 2026-01-31 (when the price was recorded)
- **Flight dates:** 2026-01-01 → 2026-01-31 (all observed flights are in January 2026)
- **Routes:** `PRG-BCN` (4,408), `PRG-AMS` (4,202), `PRG-FCO` (3,882), `PRG-LON` (3,818), `PRG-CPH` (1,501), `PRG-ATH` (1,398)
- **Split:** 15,256 `train` / 3,953 `test` rows, plus a `fold` column for cross-validation (`-1` … `4`)

### Columns

| Group | Columns | Notes |
|-------|---------|-------|
| Metadata (not used as features) | `meta_scrape_date`, `meta_flight_date`, `meta_route`, `meta_price`, `split`, `fold` | `meta_price` is the actual price in EUR |
| Target | `log_price` | Natural log of `meta_price` |
| Destination | `dest_AMS`, `dest_ATH`, `dest_BCN`, `dest_CPH`, `dest_FCO`, `dest_LON` | One-hot |
| Airline | `airline_*` (Easyjet, KLM, Ryanair, Wizz Air, Smartwings, Vueling, …) | One-hot |
| Departure period | `dep_period_morning/afternoon/evening/night`, `dep_period_ord` | One-hot + ordinal |
| Day of week | `dow_0` … `dow_6` | One-hot (0 = Monday) |
| Numeric (standardized) | `days_before`, `dom`, `holiday_dist`, `dep_minutes`, `duration_min` | Booking lead time, day of month, distance to nearest holiday, departure time, flight duration |
| Flags | `is_weekend`, `is_holiday` | Binary |

---

## Requirements

- Python 3.10+ (developed with Python 3.13)
- [PyTorch](https://pytorch.org/) (CPU build is enough)
- pandas
- numpy

```bash
pip install torch pandas numpy
```

Run every script from the project directory. They load `PRG_rep_nn_ready.csv` from a relative path, and the other scripts import `nn.py`.

---

## 1. `nn.py` — baseline neural network

```bash
python nn.py
```

What it does:

1. **`prepare_data`** loads the CSV and keeps only routes `PRG-AMS`, `PRG-FCO`, `PRG-BCN`, `PRG-LON`. It drops metadata columns and builds `X` (features) and `y` (`log_price`) tensors.
2. **`build_model`** creates a feed-forward network:
   ```
   Linear(n_features → 64) → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1)
   ```
3. **`train_model`** trains with full-batch Adam (`lr=0.01`) and MSE loss for 150 epochs.
4. **`find_optimal_flights`** predicts `log_price`, converts it back with `exp`, and picks the flight with the lowest predicted price for each route.

Output: the total parameter count and a table with `Route`, `Flight Date`, `Predicted Optimal Price`, and `Actual Price`.

> Note: `nn.py` trains and predicts on the same (historical) data, so it is an interpretation of the past rather than a forecast. Use `flight_agent.py` for future dates.

---

## 2. `interactive_interpreter.py` — explore the results

```bash
python interactive_interpreter.py
```

This script trains the `nn.py` model and then opens a menu:

1. Show all optimal flights (full table)
2. Search the optimal flight for one route (e.g. `PRG-BCN`)
3. Show potential savings (`Actual Price − Predicted Optimal Price`)
4. Exit

It must be run in an interactive terminal. If stdin is not a TTY (for example, in automated environments), it prints a hint and exits.

---

## 3. `flight_agent.py` — find the optimal future flight

The agent answers: *"If I want to fly between dates X and Y, which flight on which day should I take, and when should I buy the ticket?"*

### Quick start

```bash
# Default: flights 30–60 days from today, on PRG-AMS, PRG-FCO, PRG-BCN, PRG-LON, buying today
python flight_agent.py

# Specific travel window and routes, with a per-day price calendar
python flight_agent.py --start 2026-12-01 --end 2026-12-20 --routes PRG-BCN PRG-FCO --calendar

# Allow waiting up to 2 weeks before buying, only Ryanair and Wizz Air
python flight_agent.py --start 2026-12-01 --end 2026-12-20 --book-until 2026-11-06 --airlines Ryanair "Wizz Air"
```

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | today + 30 days | First possible flight date (`YYYY-MM-DD`) |
| `--end` | start + 30 days | Last possible flight date |
| `--today` | today | First day a ticket can be bought |
| `--book-until` | same as `--today` | Latest day you are willing to wait before buying |
| `--routes` | `PRG-AMS PRG-FCO PRG-BCN PRG-LON` | Routes to search (`PRG-CPH`, `PRG-ATH` are also in the data) |
| `--airlines` | all | Restrict to the listed airlines (case-insensitive) |
| `--top` | 3 | Number of options shown per route |
| `--calendar` | off | Also print the cheapest predicted price for each day |
| `--models` | 5 | Ensemble size (number of networks) |
| `--epochs` | 150 | Training epochs per network |
| `--data` | `PRG_rep_nn_ready.csv` | Path to the dataset |

### How it works

1. **Training (ensemble of `nn.py` networks)**
   - Uses `build_model` from `nn.py` for every network in the ensemble.
   - Trains with mini-batch Adam (batch 256, `lr=0.005`, weight decay `1e-5`) and a cosine learning-rate schedule.
   - First trains on the `train` split and reports hold-out **MAE**, **MAPE**, and a **baseline MAE** (mean price per route) on the `test` split. Then it retrains on all data for the final predictions.
   - The prediction is the mean of the ensemble. The standard deviation across networks is reported as **Uncertainty (+/-)**.

2. **Feature reconstruction (`FeatureEncoder`)**
   - The CSV contains only standardized values. The agent recovers the scaling of `days_before`, `dom`, and `holiday_dist` with a linear fit against the raw calendar values. This lets it encode features for dates that are not in the dataset.
   - Holidays: Czech public holidays, including Good Friday and Easter Monday (computed with the Gregorian Easter algorithm). The distance is capped at 15 days, as in the dataset.

3. **Flight profiles**
   - Recurring flights are extracted from the data as unique combinations of route, airline, departure time, departure period, and duration. Only flights observed at least 5 times are used.
   - The agent remembers which weekdays each flight operates on, so it only proposes flights on days they actually fly.

4. **Search**
   - Enumerates every combination of **flight date × scheduled flight × booking date** in the given window.
   - Predicts the price of each combination, ranks them, and returns the best options per route (`recommend`) or the cheapest flight per day (`price_calendar`).

### Output columns

`Route`, `Flight Date`, `Flight Weekday`, `Airline`, `Departure`, `Book On`, `Days Before`, `Predicted Price`, `Uncertainty (+/-)`, `Extrapolated`

### Using it from Python

```python
from datetime import date
from flight_agent import FlightAgent

agent = FlightAgent(n_models=5, epochs=150)
agent.train()

best = agent.recommend(date(2026, 12, 1), date(2026, 12, 20),
                       routes=['PRG-BCN'], today=date(2026, 10, 23), top=3)
print(best)

calendar = agent.price_calendar(date(2026, 12, 1), date(2026, 12, 20), routes=['PRG-BCN'])
```

### Limitations

- **Seasonality:** all training flights are in January 2026. Prices for other months are extrapolated, and the model has learned no seasonal effects. The agent prints a warning when this applies.
- **Booking lead time:** the training data covers a limited range of `days_before`. Combinations outside that range are marked `Extrapolated = True`.
- **Schedules:** flight availability is inferred from historical observations, so new or cancelled flights are not known to the agent.
- Predictions are estimates from a small model. Treat them as guidance, not as quoted prices.

*Built by  Matěj Fadrhons · [github.com/mFadr](https://github.com/mFadr)*