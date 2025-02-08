import argparse
import logging
import os
import sys

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define the DeepSVDD model
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(DeepSVDD, self).__init__()

        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2, track_running_stats=False),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 4, track_running_stats=False),

            nn.Linear(hidden_dim // 4, 64)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.center = None  # To be initialized later

    def forward(self, x):
        return self.network(x)

    def calculate_loss(self, features):
        """Calculate the SVDD loss."""
        if self.center is None:
            self.center = torch.mean(features, dim=0).detach()

        distances = torch.sum((features - self.center) ** 2, dim=1)
        loss = torch.mean(distances)
        return loss

# Define data processor
class FLDataProcessor:
    def __init__(self):
        # Update the selected_features list to match your new data columns
        self.selected_features = [
            # "src_port",
            # "dst_port",
            # "protocol",
            "bidirectional_duration_ms",
            "bidirectional_packets",
            "bidirectional_bytes",
            "src2dst_duration_ms",
            "src2dst_packets",
            "src2dst_bytes",
            "dst2src_duration_ms",
            "dst2src_packets",
            "dst2src_bytes",
            "bidirectional_min_ps",
            "bidirectional_mean_ps",
            "bidirectional_stddev_ps",
            "bidirectional_max_ps",
            "src2dst_min_ps",
            "src2dst_mean_ps",
            "src2dst_stddev_ps",
            "src2dst_max_ps",
            "dst2src_min_ps",
            "dst2src_mean_ps",
            "dst2src_stddev_ps",
            "dst2src_max_ps",
            "bidirectional_min_piat_ms",
            "bidirectional_mean_piat_ms",
            "bidirectional_stddev_piat_ms",
            "bidirectional_max_piat_ms",
            "src2dst_min_piat_ms",
            "src2dst_mean_piat_ms",
            "src2dst_stddev_piat_ms",
            "src2dst_max_piat_ms",
            "dst2src_min_piat_ms",
            "dst2src_mean_piat_ms",
            "dst2src_stddev_piat_ms",
            "dst2src_max_piat_ms",
            "bidirectional_syn_packets",
            "bidirectional_cwr_packets",
            "bidirectional_ece_packets",
            "bidirectional_ack_packets",
            "bidirectional_psh_packets",
            "bidirectional_rst_packets",
            "bidirectional_fin_packets",
            "src2dst_syn_packets",
            "src2dst_cwr_packets",
            "src2dst_ece_packets",
            "src2dst_ack_packets",
            "src2dst_psh_packets",
            "src2dst_rst_packets",
            "src2dst_fin_packets",
            "dst2src_syn_packets",
            "dst2src_ack_packets",
            "dst2src_psh_packets",
            "dst2src_fin_packets",
            "application_confidence",
            "udps.session_establishment_request_counter",
            "udps.session_deletion_request_counter",
            "udps.session_report_request_counter",
        ]

    def preprocess_data(self, data_path_1, data_path_2, test_size=0.15, val_size=0.15):
        """Load and preprocess data streams with training, validation, and test splits"""
        # Load datasets
        df1 = pd.read_csv(data_path_1, nrows=30000)
        df2 = pd.read_csv(data_path_2, nrows=30000)

        print(f"\nSamples in dataset 1: {len(df1)}")
        print(f"Samples in dataset 2: {len(df2)}")

        # Shuffle the data to ensure random distribution
        df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)
        df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)

        # Keep selected features; exclude labels for real-time training
        features_to_keep = self.selected_features  # Exclude 'Label'
        df1 = df1[features_to_keep]
        df2 = df2[features_to_keep]

        # Split into features
        X1 = df1[self.selected_features].values
        X2 = df2[self.selected_features].values

        # For real-time FL, we don't need labels during training
        # Labels can be used separately for offline evaluation

        # Split each client's data into train, validation, and test sets
        X1_train, X1_temp = train_test_split(
            X1, test_size=(val_size + test_size), random_state=42)
        X1_val, X1_test = train_test_split(
            X1_temp, test_size=test_size/(val_size + test_size),
            random_state=42)

        X2_train, X2_temp = train_test_split(
            X2, test_size=(val_size + test_size), random_state=42)
        X2_val, X2_test = train_test_split(
            X2_temp, test_size=test_size/(val_size + test_size),
            random_state=42)

        # Print data statistics
        self._print_statistics("Client 1 - Train", X1_train)
        self._print_statistics("Client 1 - Validation", X1_val)
        self._print_statistics("Client 1 - Test", X1_test)

        self._print_statistics("Client 2 - Train", X2_train)
        self._print_statistics("Client 2 - Validation", X2_val)
        self._print_statistics("Client 2 - Test", X2_test)

        return {
            'train': [X1_train, X2_train],
            'validation': [X1_val, X2_val],
            'test': [X1_test, X2_test]
        }

    def _print_statistics(self, name, data):
        total = len(data)
        print(f"\n{name} Statistics:")
        print(f"Total samples: {total}")

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, optimizer, scaler, x_train, client_id, fraction_fit=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.client_id = client_id
        self.fraction_fit = fraction_fit

        # Prepare dataset (unsupervised: targets are same as inputs)
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)

    def get_parameters(self):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        state_dict = {}
        for key, val in zip(self.model.state_dict().keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data and return updated parameters."""
        self.set_parameters(parameters)

        self.model.train()
        for epoch in range(1):  # One epoch per Flower round
            for batch in self.loader:
                x = batch[0].to(self.model.device)
                self.optimizer.zero_grad()
                features = self.model(x)
                loss = self.model.calculate_loss(features)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local test data."""
        self.set_parameters(parameters)
        # Evaluation is handled separately; return dummy values
        return 0.0, len(self.loader.dataset), {}

def run_client(args):
    """Function to run a Flower client."""
    logger.info(f"Starting Flower client {args.client_id}...")

    # Initialize data processor
    processor = FLDataProcessor()
    data_splits = processor.preprocess_data(args.data_path_1, args.data_path_2)
    X_train = data_splits['train'][args.client_id - 1]  # Assuming client_id starts at 1

    # Initialize scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    input_dim = X_train.shape[1]

    # Initialize model
    model = DeepSVDD(input_dim=input_dim)
    model = model.to(model.device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    # Create Flower client instance
    client = FlowerClient(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        x_train=X_train,
        client_id=args.client_id,
        fraction_fit=args.fraction_fit
    )

    # Start Flower client using the updated method
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )

def main():
    parser = argparse.ArgumentParser(description='Real-Time Federated Learning with Flower and DeepSVDD')
    parser.add_argument('--client', type=str, help='Run as Flower client and specify server address')
    parser.add_argument('--capacity', type=int, default=800, help='Buffer capacity')
    parser.add_argument('--buffer_size', type=int, default=800, help='Buffer size')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, help='Anomaly threshold')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated learning rounds')
    parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients used in each round')
    parser.add_argument('--input_dim', type=int, default=49, help='Input dimension for the model')
    parser.add_argument('--client_id', type=int, default=1, help='Unique client identifier')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    parser.add_argument('--port', type=int, default=7000, help='Port number for server or client')
    parser.add_argument('--server_address', type=str, default='nwdaf-0:7000',
                        help='Flower server address (for clients)')
    parser.add_argument('--data_path_1', type=str, default='data/slice_1.csv',
                        help='Path to the first data CSV file (for clients)')
    parser.add_argument('--data_path_2', type=str, default='data/slice_2.csv',
                        help='Path to the second data CSV file (for clients)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')

    args = parser.parse_args()

    set_seed(42)

    if args.client:
        run_client(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
