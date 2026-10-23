import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------- #
# 	Loads the dataset, filters for the specified
# 	routes, separates features from targets, and
# 	converts them into PyTorch tensors for network
# 	processing.
# ---------------------------------------------- #
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    target_routes = ['PRG-AMS', 'PRG-FCO', 'PRG-BCN', 'PRG-LON']
    df = df[df['meta_route'].isin(target_routes)].copy()

    exclude_cols = ['meta_scrape_date', 'meta_flight_date', 'meta_route', 'meta_price', 'split', 'fold', 'log_price']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df['log_price'].values, dtype=torch.float32).view(-1, 1)

    return X, y, df

# ---------------------------------------------- #
# 	Defines the feedforward neural network using
# 	PyTorch's Sequential container. Employs Linear
# 	layers and ReLU activations for continuous
# 	price prediction.
# ---------------------------------------------- #
def build_model(input_size):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model

# ---------------------------------------------- #
# 	Trains the neural network utilizing Mean Squared
# 	Error loss and the Adam optimizer. Iterates over
# 	the dataset to update model weights via
# 	backpropagation.
# ---------------------------------------------- #
def train_model(model, X, y, epochs=150, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# ---------------------------------------------- #
# 	Utilizes the trained model to predict log_prices
# 	and converts them to absolute prices. Groups
# 	predictions by route to pinpoint the optimal
# 	(cheapest) flight per route and structures
# 	the final interpretation table.
# ---------------------------------------------- #
def find_optimal_flights(model, X, df):
    with torch.no_grad():
        predictions = model(X).numpy()

    df['predicted_log_price'] = predictions
    df['predicted_price'] = np.exp(predictions)

    optimal_idx = df.groupby('meta_route')['predicted_price'].idxmin()
    optimal_flights = df.loc[optimal_idx]

    interpretation_table = optimal_flights[['meta_route', 'meta_flight_date', 'predicted_price', 'meta_price']]
    interpretation_table = interpretation_table.rename(columns={
        'meta_route': 'Route',
        'meta_flight_date': 'Flight Date',
        'predicted_price': 'Predicted Optimal Price',
        'meta_price': 'Actual Price'
    })

    return interpretation_table

# ---------------------------------------------- #
# 	Main execution sequence orchestrating data
# 	loading, model initialization, evaluation, and
# 	printing parameter counts using numel alongside
# 	the final prediction table.
# ---------------------------------------------- #
def main():
    file_path = 'PRG_rep_nn_ready.csv'
    X, y, df = prepare_data(file_path)

    input_size = X.shape[1]
    model = build_model(input_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Neural Network Initialized. Total parameters: {total_params}\n")

    trained_model = train_model(model, X, y, epochs=150)

    results_table = find_optimal_flights(trained_model, X, df)
    print("--- OPTIMAL FLIGHT PREDICTIONS INTERPRETATION TABLE ---")
    print(results_table.to_string(index=False))

if __name__ == '__main__':
    main()