# client_plugin.py
import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, List

# -----------------------
# 1) Advanced OnlineDeepSVDD Class
# -----------------------
class OnlineDeepSVDD(nn.Module):
    """
    Full advanced partial-fit SVDD with center/radius logic, warmup,
    distance history, partial FedProx example, etc.
    """

    def __init__(
        self,
        input_dim=49,
        hidden_dim=512,
        nu=0.5,
        warmup_batches=1500,
        window_size=8000,
        fedprox_mu=0.07,
        ema_alpha=0.95,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm1d(hidden_dim // 2, track_running_stats=False),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm1d(hidden_dim // 4, track_running_stats=False),

            nn.Linear(hidden_dim // 4, 64)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nu = nu
        self.warmup_batches = warmup_batches
        self.window_size = window_size
        self.fedprox_mu = fedprox_mu
        self.ema_alpha = ema_alpha

        # SVDD parameters
        self.center = None
        self.radius = None
        self.min_radius = None
        self.distance_history = []
        self.current_batch = 0

        # For partial fit: keep track of global model (FedProx example)
        self.global_model_params = None

        # For EMA (exponential moving average)
        self.ema_center = None
        self.ema_radius = None

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def update_ema_parameters(self):
        """Exponential moving average of center, radius."""
        if self.ema_center is None and self.center is not None:
            self.ema_center = self.center.clone()
        elif self.center is not None:
            self.ema_center = (
                self.ema_alpha * self.ema_center.clone()
                + (1 - self.ema_alpha) * self.center.clone()
            ).detach()

        if self.radius is not None:
            if self.ema_radius is None:
                self.ema_radius = self.radius.clone()
            else:
                self.ema_radius = (
                    self.ema_alpha * self.ema_radius.clone()
                    + (1 - self.ema_alpha) * self.radius.clone()
                ).detach()

    def partial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning features for SVDD loss."""
        return self.forward(x)

    def calculate_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Example partial SVDD loss with warmups, radius, etc.
        """
        # Warmup phase: just average distance
        if self.current_batch < self.warmup_batches:
            return torch.mean(torch.sum((features - self.center_init(features))**2, dim=1))

        # If not in warmup
        distances = torch.sum((features - self.center_init(features)) ** 2, dim=1)
        self.distance_history.extend(distances.detach().cpu().numpy())

        # Maintain a rolling window of distances
        if len(self.distance_history) > self.window_size:
            self.distance_history = self.distance_history[-self.window_size:]

        # radius = quantile(1 - nu)
        current_radius = np.quantile(self.distance_history, 1 - self.nu)

        if self.min_radius is None:
            self.min_radius = current_radius
        else:
            self.min_radius = min(self.min_radius, current_radius)

        target_radius = max(current_radius, self.min_radius)
        new_radius = torch.tensor(target_radius, device=self.device)

        if self.radius is None:
            self.radius = new_radius
        else:
            # Weighted update
            self.radius = 0.95 * self.radius + 0.05 * new_radius
            self.radius = self.radius.detach()

        # Basic SVDD: L = mean(max(0, dist^2 - R^2)) plus reg
        zeros = torch.zeros_like(distances)
        svdd_loss = torch.mean(torch.maximum(zeros, distances - self.radius))

        # Additional center penalty or L2, FedProx, etc.
        l2_reg = 0.001 * sum(torch.sum(param**2) for param in self.parameters())

        # FedProx example if we have global params
        fedprox_term = 0.0
        if self.global_model_params is not None:
            for p_local, p_global in zip(self.parameters(), self.global_model_params):
                fedprox_term += self.fedprox_mu * torch.sum((p_local - p_global)**2)

        loss = svdd_loss + l2_reg + fedprox_term
        return loss

    def center_init(self, features: torch.Tensor) -> torch.Tensor:
        """Initialize or use center with EMA."""
        if self.center is None:
            # initial center = mean of features
            self.center = torch.mean(features, dim=0).detach()
        return self.center


# -----------------------
# 2) The Flower NumPyClient that uses OnlineDeepSVDD
# -----------------------
class SVDDNumPyClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: OnlineDeepSVDD,
        x_data: np.ndarray,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create dataset / dataloader
        dataset = TensorDataset(torch.from_numpy(x_data).float())
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.dataset_size = len(dataset)

    def get_parameters(self) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy ndarrays."""
        return [p.cpu().detach().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load new parameters into local model."""
        keys = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        """Train on local data for one round."""
        self.set_parameters(parameters)

        # Possibly store global params for FedProx
        self.model.global_model_params = [
            p.detach().clone() for p in self.model.parameters()
        ]

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # One epoch example
        for _ in range(1):
            for x_batch in self.loader:
                x = x_batch[0].to(self.model.device)
                optimizer.zero_grad()
                feats = self.model.partial_forward(x)
                loss = self.model.calculate_loss(feats)
                loss.backward()
                optimizer.step()

            # after each epoch/batch increment
            self.model.current_batch += 1
            self.model.update_ema_parameters()

        return self.get_parameters(), self.dataset_size, {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict]:
        """Evaluate on local data (dummy)."""
        self.set_parameters(parameters)
        # Optionally compute average distance or radius
        # We'll just return 0.0 for demonstration
        return 0.0, self.dataset_size, {}


# -----------------------
# 3) The plugin factory for flower-supernode
# -----------------------
def client_factory(args: Dict[str, Any]) -> fl.client.Client:
    """
    Called by: flower-supernode client_plugin.py --args='{"csv_path_1":"...","csv_path_2":"...","warmup_batches":1500, ...}'
    Returns a NumPyClient instance that uses OnlineDeepSVDD with advanced features.
    """
    # Extract arguments from CLI
    csv_path_1 = args.get("csv_path_1", "data/slice_1.csv")
    csv_path_2 = args.get("csv_path_2", "data/slice_2.csv")
    input_dim = int(args.get("input_dim", 49))
    hidden_dim = int(args.get("hidden_dim", 512))
    nu = float(args.get("nu", 0.5))
    warmup_batches = int(args.get("warmup_batches", 1500))
    window_size = int(args.get("window_size", 8000))
    fedprox_mu = float(args.get("fedprox_mu", 0.07))
    ema_alpha = float(args.get("ema_alpha", 0.95))
    learning_rate = float(args.get("learning_rate", 1e-3))
    batch_size = int(args.get("batch_size", 32))
    samples = int(args.get("samples", 30000))

    # Load CSV data
    x_data = load_csv_data(csv_path_1, csv_path_2, input_dim, samples)

    # Initialize advanced OnlineDeepSVDD model
    svdd_model = OnlineDeepSVDD(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nu=nu,
        warmup_batches=warmup_batches,
        window_size=window_size,
        fedprox_mu=fedprox_mu,
        ema_alpha=ema_alpha,
    )

    # Build the Flower client
    return SVDDNumPyClient(
        model=svdd_model,
        x_data=x_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def load_csv_data(path1: str, path2: str, input_dim: int, samples: int) -> np.ndarray:
    """Load data from two CSVs, combine, return NxD array."""
    try:
        df1 = pd.read_csv(path1, nrows=samples)
        df2 = pd.read_csv(path2, nrows=samples)
        # Retain first input_dim columns as features
        arr1 = df1.iloc[:, :input_dim].values
        arr2 = df2.iloc[:, :input_dim].values
        combined = np.concatenate([arr1, arr2], axis=0)
        np.random.shuffle(combined)
        return combined.astype(np.float32)
    except Exception as e:
        print(f"Error loading CSV data: {e}. Using dummy fallback.")
        return np.random.randn(samples, input_dim).astype(np.float32)
