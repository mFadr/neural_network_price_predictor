import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
import torch.optim as optim

# The base network architecture is reused from nn.py
from nn import build_model

DATA_FILE = 'PRG_rep_nn_ready.csv'
DEFAULT_ROUTES = ['PRG-AMS', 'PRG-FCO', 'PRG-BCN', 'PRG-LON']
EXCLUDE_COLS = ['meta_scrape_date', 'meta_flight_date', 'meta_route', 'meta_price', 'split', 'fold', 'log_price']

# Holidays as encoded in the dataset (1 Jan and 6 Jan are flagged), extended
# with the Czech public holidays so dates outside January get a sensible value.
FIXED_HOLIDAYS = [(1, 1), (1, 6), (5, 1), (5, 8), (7, 5), (7, 6), (9, 28), (10, 28), (11, 17), (12, 24), (12, 25), (12, 26)]
HOLIDAY_DIST_CAP = 15

# Minimum number of observations for a (route, airline, departure) profile to
# be treated as a regularly scheduled flight.
MIN_PROFILE_COUNT = 5


# ---------------------------------------------- #
# 	Returns the Easter Sunday date for a given
# 	year (anonymous Gregorian algorithm). Used for
# 	the moving Czech holidays (Good Friday and
# 	Easter Monday).
# ---------------------------------------------- #
def easter_sunday(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def holidays_for_year(year):
    days = {date(year, m, d) for m, d in FIXED_HOLIDAYS}
    easter = easter_sunday(year)
    days.add(easter - timedelta(days=2))
    days.add(easter + timedelta(days=1))
    return days


# ---------------------------------------------- #
# 	Distance in days to the nearest holiday, capped
# 	the same way as in the dataset.
# ---------------------------------------------- #
def holiday_distance(day):
    holidays = holidays_for_year(day.year - 1) | holidays_for_year(day.year) | holidays_for_year(day.year + 1)
    dist = min(abs((day - h).days) for h in holidays)
    return min(dist, HOLIDAY_DIST_CAP), int(dist == 0)


# ---------------------------------------------- #
# 	Converts raw calendar values into the
# 	standardized features the network was trained
# 	on. The scaling (mean/std) of each column is
# 	recovered from the dataset with a linear fit,
# 	because the CSV only contains scaled values.
# ---------------------------------------------- #
class FeatureEncoder:
    def __init__(self, df, feature_cols):
        self.feature_cols = feature_cols
        flight = pd.to_datetime(df['meta_flight_date'])
        scrape = pd.to_datetime(df['meta_scrape_date'])

        raw_days_before = (flight - scrape).dt.days.values
        raw_dom = flight.dt.day.values
        raw_hol = np.array([holiday_distance(d.date())[0] for d in flight])

        self.scalers = {}
        for col, raw in [('days_before', raw_days_before), ('dom', raw_dom), ('holiday_dist', raw_hol)]:
            slope, intercept = np.polyfit(raw, df[col].values, 1)
            max_err = np.abs(slope * raw + intercept - df[col].values).max()
            if max_err > 1e-3:
                raise ValueError(f"Could not reconstruct scaling of '{col}' (max error {max_err:.4f}).")
            self.scalers[col] = (slope, intercept)

        self.days_before_range = (int(raw_days_before.min()), int(raw_days_before.max()))
        self.train_months = sorted(int(m) for m in flight.dt.month.unique())

    def scale(self, col, raw):
        slope, intercept = self.scalers[col]
        return slope * raw + intercept

    # Builds one feature row for a flight profile on a given flight date,
    # booked on a given booking date.
    def encode(self, profile, flight_date, booking_date):
        row = dict(profile)
        weekday = flight_date.weekday()
        for i in range(7):
            row[f'dow_{i}'] = float(i == weekday)
        hol_dist, is_hol = holiday_distance(flight_date)
        row['days_before'] = self.scale('days_before', (flight_date - booking_date).days)
        row['dom'] = self.scale('dom', flight_date.day)
        row['holiday_dist'] = self.scale('holiday_dist', hol_dist)
        row['is_weekend'] = float(weekday >= 5)
        row['is_holiday'] = float(is_hol)
        return [row[c] for c in self.feature_cols]


# ---------------------------------------------- #
# 	Agent that learns flight prices with the nn.py
# 	network and searches the future (flight date,
# 	flight, booking date) space for the cheapest
# 	combination per route.
# ---------------------------------------------- #
class FlightAgent:
    def __init__(self, file_path=DATA_FILE, n_models=5, epochs=150, lr=0.005, seed=42):
        self.file_path = file_path
        self.n_models = n_models
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.models = []
        self.metrics = {}

        self.df = pd.read_csv(file_path)
        self.feature_cols = [c for c in self.df.columns if c not in EXCLUDE_COLS]
        self.encoder = FeatureEncoder(self.df, self.feature_cols)
        self.profiles = self._extract_profiles()

    # Every recurring flight (route + airline + departure time + duration)
    # together with the weekdays it was observed to operate on.
    def _extract_profiles(self):
        static_cols = [c for c in self.feature_cols
                       if c.startswith(('dest_', 'airline_', 'dep_period_'))
                       or c in ('dep_minutes', 'duration_min')]
        df = self.df.copy()
        df['weekday'] = pd.to_datetime(df['meta_flight_date']).dt.weekday
        df['airline'] = df[[c for c in static_cols if c.startswith('airline_')]].idxmax(axis=1).str.replace('airline_', '')

        grouped = df.groupby(['meta_route'] + static_cols)
        profiles = []
        for key, group in grouped:
            if len(group) < MIN_PROFILE_COUNT:
                continue
            values = dict(zip(['meta_route'] + static_cols, key))
            profiles.append({
                'route': values.pop('meta_route'),
                'airline': group['airline'].iloc[0],
                'period': next(c.replace('dep_period_', '') for c in static_cols
                               if c.startswith('dep_period_') and c != 'dep_period_ord' and values[c] == 1.0),
                'weekdays': set(group['weekday'].unique()),
                'features': values,
                'n_obs': len(group),
            })
        return profiles

    def _tensors(self, df):
        X = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df['log_price'].values, dtype=torch.float32).view(-1, 1)
        return X, y

    # Mini-batch training of a single nn.py network.
    def _train_one(self, X, y, seed):
        torch.manual_seed(seed)
        model = build_model(X.shape[1])
        criterion = tnn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        batch_size = 256

        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X[idx]), y[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()
        model.eval()
        return model

    def _predict_log(self, X):
        with torch.no_grad():
            preds = torch.stack([m(X) for m in self.models]).squeeze(-1).numpy()
        return preds  # shape (n_models, n_rows)

    # Trains the ensemble on the 'train' split, reports the error on the
    # 'test' split, then retrains on all rows for the final agent.
    def train(self, verbose=True):
        train_df = self.df[self.df['split'] == 'train']
        test_df = self.df[self.df['split'] == 'test']

        if verbose:
            print(f"Training {self.n_models} network(s) on {len(train_df)} rows, evaluating on {len(test_df)} rows...")
        X_tr, y_tr = self._tensors(train_df)
        X_te, _ = self._tensors(test_df)
        self.models = [self._train_one(X_tr, y_tr, self.seed + i) for i in range(self.n_models)]

        pred = np.exp(self._predict_log(X_te).mean(axis=0))
        actual = test_df['meta_price'].values
        self.metrics = {
            'MAE (EUR)': float(np.mean(np.abs(pred - actual))),
            'MAPE (%)': float(np.mean(np.abs(pred - actual) / actual) * 100),
        }
        baseline = np.exp(train_df.groupby('meta_route')['log_price'].mean())
        base_pred = test_df['meta_route'].map(baseline).values
        self.metrics['Baseline MAE (EUR)'] = float(np.mean(np.abs(base_pred - actual)))

        if verbose:
            print("Hold-out test: " + ", ".join(f"{k}={v:.2f}" for k, v in self.metrics.items()))
            print("Retraining on the full dataset...")
        X_all, y_all = self._tensors(self.df)
        self.models = [self._train_one(X_all, y_all, self.seed + 100 + i) for i in range(self.n_models)]
        return self.metrics

    # ---------------------------------------------- #
    # 	Core search: enumerates every combination of
    # 	flight date x scheduled flight x booking date,
    # 	predicts its price and ranks them.
    # 	- today: the first day a ticket can be bought
    # 	- book_until: last day you are willing to wait
    # 	  with buying (None = buy today)
    # ---------------------------------------------- #
    def search(self, start, end, routes=None, today=None, book_until=None, airlines=None, weekdays=None):
        if not self.models:
            raise RuntimeError("Agent is not trained. Call train() first.")
        routes = routes or DEFAULT_ROUTES
        today = today or date.today()
        book_until = book_until or today
        if start < today:
            raise ValueError("Flight window must start today or later.")

        lo, hi = self.encoder.days_before_range
        rows, meta = [], []
        flight_date = start
        while flight_date <= end:
            if weekdays is None or flight_date.weekday() in weekdays:
                booking_dates = [today + timedelta(days=i) for i in range((min(book_until, flight_date) - today).days + 1)]
                for prof in self.profiles:
                    if prof['route'] not in routes or flight_date.weekday() not in prof['weekdays']:
                        continue
                    if airlines and prof['airline'].lower() not in airlines:
                        continue
                    for booking in booking_dates:
                        days_before = (flight_date - booking).days
                        rows.append(self.encoder.encode(prof['features'], flight_date, booking))
                        meta.append((prof['route'], flight_date, booking, days_before, prof['airline'], prof['period'],
                                     prof['features']['dep_minutes'], prof['n_obs'],
                                     not (lo <= days_before <= hi)))
            flight_date += timedelta(days=1)

        if not rows:
            return pd.DataFrame()

        preds = self._predict_log(torch.tensor(np.array(rows), dtype=torch.float32))
        prices = np.exp(preds)
        result = pd.DataFrame(meta, columns=['Route', 'Flight Date', 'Book On', 'Days Before', 'Airline', 'Departure',
                                             'dep_minutes_scaled', 'Observations', 'Extrapolated'])
        result['Predicted Price'] = prices.mean(axis=0).astype(np.float64)
        result['Uncertainty (+/-)'] = prices.std(axis=0).astype(np.float64)
        result['Flight Weekday'] = pd.to_datetime(result['Flight Date']).dt.day_name()
        return result.sort_values('Predicted Price').reset_index(drop=True)

    # Cheapest combination per route.
    def recommend(self, *args, top=1, **kwargs):
        result = self.search(*args, **kwargs)
        if result.empty:
            return result
        # Keep only the best booking day of each distinct flight
        result = result.drop_duplicates(['Route', 'Flight Date', 'Airline', 'dep_minutes_scaled'])
        best = result.groupby('Route', group_keys=False).head(top)
        cols = ['Route', 'Flight Date', 'Flight Weekday', 'Airline', 'Departure', 'Book On', 'Days Before',
                'Predicted Price', 'Uncertainty (+/-)', 'Extrapolated']
        return best[cols].sort_values(['Route', 'Predicted Price']).reset_index(drop=True)

    # Cheapest predicted price per route and flight date (a price calendar).
    def price_calendar(self, *args, **kwargs):
        result = self.search(*args, **kwargs)
        if result.empty:
            return result
        idx = result.groupby(['Route', 'Flight Date'])['Predicted Price'].idxmin()
        cal = result.loc[idx, ['Route', 'Flight Date', 'Flight Weekday', 'Airline', 'Departure', 'Book On', 'Predicted Price']]
        return cal.sort_values(['Route', 'Flight Date']).reset_index(drop=True)

    def warnings_for(self, start, end):
        notes = []
        months = {d.month for d in pd.date_range(start, end)}
        if not months.issubset(self.encoder.train_months):
            notes.append("The training data only contains flights in month(s) "
                         f"{self.encoder.train_months}; prices for other months are extrapolated "
                         "(no seasonal effects are learned).")
        lo, hi = self.encoder.days_before_range
        notes.append(f"Booking lead times outside {lo}-{hi} days are marked as 'Extrapolated'.")
        return notes


def parse_date(text):
    return date.fromisoformat(text)


def build_parser():
    parser = argparse.ArgumentParser(description="Agent that finds the optimal future flight date and flight from Prague.")
    parser.add_argument('--start', type=parse_date, help="First possible flight date (YYYY-MM-DD).")
    parser.add_argument('--end', type=parse_date, help="Last possible flight date (YYYY-MM-DD).")
    parser.add_argument('--today', type=parse_date, default=date.today(), help="Booking date (default: today).")
    parser.add_argument('--book-until', type=parse_date, help="Latest date you are willing to wait before buying.")
    parser.add_argument('--routes', nargs='+', default=DEFAULT_ROUTES, help="Routes, e.g. PRG-AMS PRG-BCN.")
    parser.add_argument('--airlines', nargs='+', help="Restrict to these airlines.")
    parser.add_argument('--top', type=int, default=3, help="Number of options shown per route.")
    parser.add_argument('--calendar', action='store_true', help="Also print the cheapest price per day.")
    parser.add_argument('--models', type=int, default=5, help="Ensemble size.")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--data', default=DATA_FILE)
    return parser


def main():
    args = build_parser().parse_args()
    start = args.start or args.today + timedelta(days=30)
    end = args.end or start + timedelta(days=30)
    routes = [r.upper() for r in args.routes]
    airlines = [a.lower() for a in args.airlines] if args.airlines else None

    agent = FlightAgent(args.data, n_models=args.models, epochs=args.epochs)
    agent.train()

    print(f"\nSearching flights {start} -> {end}, buying between {args.today} and {args.book_until or args.today}")
    for note in agent.warnings_for(start, end):
        print(f"  NOTE: {note}")

    pd.set_option('display.width', 200)
    best = agent.recommend(start, end, routes=routes, today=args.today, book_until=args.book_until,
                           airlines=airlines, top=args.top)
    if best.empty:
        print("No scheduled flights match the given criteria.")
        sys.exit(0)
    print("\n--- OPTIMAL FLIGHT RECOMMENDATIONS ---")
    print(best.round(2).to_string(index=False))

    if args.calendar:
        cal = agent.price_calendar(start, end, routes=routes, today=args.today,
                                   book_until=args.book_until, airlines=airlines)
        print("\n--- PRICE CALENDAR (cheapest per day) ---")
        print(cal.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
