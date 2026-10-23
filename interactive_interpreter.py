import pandas as pd
import sys
import os

# Import functions from the previous script
try:
    from nn import prepare_data, build_model, train_model, find_optimal_flights
except ImportError:
    print("Error: Could not import nn.py. Make sure it is in the same directory.")
    sys.exit(1)

# ---------------------------------------------- #
# 	description: Retrieves the optimal flight
# 	table by running the models from the
# 	imported previous script.
# ---------------------------------------------- #
def get_results_table():
    print("Loading data and training model... Please wait.")
    file_path = 'PRG_rep_nn_ready.csv'

    try:
        X, y, df = prepare_data(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the dataset '{file_path}'.")
        sys.exit(1)

    input_size = X.shape[1]
    model = build_model(input_size)
    trained_model = train_model(model, X, y, epochs=150)

    results_table = find_optimal_flights(trained_model, X, df)
    return results_table

# ---------------------------------------------- #
# 	description: Main loop for the interactive
# 	CLI to explore the neural network's flight
# 	predictions. Uses standard input to interact.
# ---------------------------------------------- #
def interactive_menu():
    table = get_results_table()

    while True:
        print("\n" + "="*50)
        print("  INTERACTIVE OPTIMAL FLIGHT INTERPRETER  ")
        print("="*50)
        print("1. Show all optimal flights (Full Table)")
        print("2. Search for optimal flight by Route")
        print("3. Show potential savings (Actual vs Predicted)")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            print("\n--- FULL OPTIMAL FLIGHT PREDICTIONS ---")
            print(table.to_string(index=False))

        elif choice == '2':
            route = input("Enter route (e.g., PRG-AMS, PRG-BCN, PRG-FCO, PRG-LON): ").strip().upper()
            route_data = table[table['Route'] == route]
            if not route_data.empty:
                print(f"\n--- OPTIMAL FLIGHT FOR {route} ---")
                print(route_data.to_string(index=False))
            else:
                print(f"\nNo data found for route: {route}. Please check your input.")

        elif choice == '3':
            print("\n--- POTENTIAL SAVINGS ---")
            savings_table = table.copy()
            # Calculate difference to see how much the predicted optimal is compared to actual
            savings_table['Difference'] = savings_table['Actual Price'] - savings_table['Predicted Optimal Price']
            print(savings_table[['Route', 'Actual Price', 'Predicted Optimal Price', 'Difference']].to_string(index=False))

        elif choice == '4':
            print("Exiting interactive interpreter. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == '__main__':
    # Checking if running in a terminal to avoid blocking automated environments
    if not os.isatty(sys.stdin.fileno()):
        print("Script generated successfully. Run 'python interactive_interpreter.py' in a terminal to use the interactive menu.")
    else:
        interactive_menu()