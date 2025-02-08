import flwr as fl

def main():
    # Define server strategy without eval_fn
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        # Removed eval_fn to align with Flower 1.1.0
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:7000",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
