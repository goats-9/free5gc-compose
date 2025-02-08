#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------
# 1) Utility for reproducibility
# -----------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------
# 2) DeepSVDD network
# -----------------------------------------------------
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(DeepSVDD, self).__init__()
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

    def forward(self, x):
        return self.network(x)

# -----------------------------------------------------
# 3) Advanced partial-fitting SVDD
# -----------------------------------------------------
class OnlineDeepSVDD:
    """
    Unsupervised advanced SVDD with partial-fitting logic:
    - center/radius updates
    - warmup
    - FedProx support (global_model_params)
    - no label usage in training
    """

    def __init__(
        self,
        input_dim,
        batch_size=800,
        learning_rate=0.001,
        nu=0.5,
        warmup_batches=1500,
        window_size=4000,
        fedprox_mu=0.01,
        ema_alpha=0.95,
        prediction_threshold=0.2,
        smooth_window=10
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = DeepSVDD(input_dim).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Basic partial-fitting
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.is_initialized = False
        
        # SVDD
        self.center = None
        self.radius = None
        self.min_radius = None
        self.distance_history = []
        self.current_batch = 0

        self.nu = nu
        self.warmup_batches = warmup_batches
        self.window_size = window_size

        # FedProx
        self.global_model_params = None
        self.fedprox_mu = fedprox_mu

        # EMA
        self.ema_alpha = ema_alpha
        self.ema_center = None
        self.ema_radius = None

        # Smoothing
        self.prediction_threshold = prediction_threshold
        self.smooth_window = smooth_window
        self.previous_predictions = []

        # Thread lock
        self.lock = threading.Lock()

        # Stats
        self.recent_f1_scores = []
        self.performance_window = 5

    def initialize_center(self, X, eps=0.02):
        """Initialize center from first batch."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            feats = self.model(X_tensor)
            c = torch.mean(feats, dim=0)
            noise = eps * torch.randn_like(c)
            self.center = (c + noise).clone().detach()
            self.ema_center = self.center.clone()

    def update_ema_parameters(self):
        if self.ema_center is None:
            self.ema_center = self.center.clone()
        else:
            self.ema_center = (
                self.ema_alpha * self.ema_center + (1 - self.ema_alpha) * self.center
            ).detach()

        if self.radius is not None:
            if self.ema_radius is None:
                self.ema_radius = self.radius.clone()
            else:
                self.ema_radius = (
                    self.ema_alpha * self.ema_radius + (1 - self.ema_alpha) * self.radius
                ).detach()

    def calculate_loss(self, distances, feats):
        """Compute partial-fitting SVDD loss."""
        if self.current_batch < self.warmup_batches:
            # warmup => average distance
            return torch.mean(distances) + 0.01*torch.mean(torch.abs(feats))

        # track distances
        current_dist = distances.detach().cpu().numpy()
        self.distance_history.extend(current_dist)
        if len(self.distance_history) > self.window_size:
            self.distance_history = self.distance_history[-self.window_size:]

        # compute radius
        import numpy as np
        current_radius = np.quantile(self.distance_history, 1 - self.nu)
        if self.min_radius is None:
            self.min_radius = current_radius
        else:
            self.min_radius = min(self.min_radius, current_radius)

        target_radius = max(current_radius, self.min_radius)
        target_radius_t = torch.tensor(target_radius, device=self.device)

        if self.radius is None:
            self.radius = target_radius_t
        else:
            self.radius = 0.95*self.radius + 0.05*target_radius_t
            self.radius = self.radius.detach()

        # SVDD
        zeros = torch.zeros_like(distances)
        svdd_loss = torch.mean(torch.maximum(zeros, distances - self.radius))

        # small L2 reg
        l2_reg = 0.001 * sum(torch.sum(p.pow(2)) for p in self.model.parameters())

        # center penalty
        center_penalty = 0.01 * torch.sqrt(torch.sum(self.center**2))

        loss = svdd_loss + l2_reg + center_penalty

        # FedProx
        if self.global_model_params is not None:
            fedprox_term = 0.0
            for p_local, p_global in zip(self.model.parameters(), self.global_model_params):
                fedprox_term += self.fedprox_mu * torch.sum((p_local - p_global)**2)
            loss += fedprox_term

        return loss

    def partial_fit(self, X):
        """Partial-fitting on one batch X."""
        with self.lock:
            if not self.is_initialized:
                # init
                self.scaler.partial_fit(X)
                self.initialize_center(X)
                self.is_initialized = True
                return

            try:
                # partial fit
                self.scaler.partial_fit(X)
                X_tensor = torch.FloatTensor(self.scaler.transform(X)).to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                feats = self.model(X_tensor)
                dist = torch.sum((feats - self.center.detach())**2, dim=1)
                loss = self.calculate_loss(dist, feats)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                self.current_batch += 1
                with torch.no_grad():
                    self.update_ema_parameters()

                return loss.item()

            except Exception as e:
                print("Error in partial_fit:", e)
                raise e

    def predict(self, X):
        """Predict anomalies (1=anomaly,0=benign) with smoothing."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            feats = self.model(X_t)
            center = self.ema_center if self.ema_center is not None else self.center
            dist = torch.sum((feats - center)**2,dim=1)
            if self.radius is None:
                # fallback
                cur_r = torch.quantile(dist,1-self.nu)
                self.radius = cur_r.clone().detach()
            rad = self.ema_radius if self.ema_radius is not None else self.radius

            raw_preds = (dist>rad).float().cpu().numpy()
            self.previous_predictions.append(raw_preds)
            if len(self.previous_predictions)>self.smooth_window:
                self.previous_predictions.pop(0)

            smoothed = np.mean(self.previous_predictions, axis=0)
            final = (smoothed>self.prediction_threshold).astype(int)
            return final

    def get_performance_weight(self):
        """Optional weighting for aggregator."""
        # We'll just return 0.5 if no real F1 tracking
        if len(self.recent_f1_scores)>0:
            import numpy as np
            ws=np.exp(np.linspace(-1,0,len(self.recent_f1_scores)))
            ws/=ws.sum()
            sc=np.sum(ws*self.recent_f1_scores)
            return max(0.1,min(sc,1.0))
        return 0.5

    def update_performance_score(self, f1_score):
        """Track recent F1 (not used for training, but aggregator weighting)."""
        self.recent_f1_scores.append(f1_score)
        if len(self.recent_f1_scores)>self.performance_window:
            self.recent_f1_scores.pop(0)

# -----------------------------------------------------
# 4) Federated aggregator
# -----------------------------------------------------
class FederatedServer:
    """
    Runs merges on an interval in background.
    Weighted average logic by client performance weighting.
    """
    def __init__(self, clients, federation_interval=70):
        self.clients = clients
        self.federation_interval = federation_interval
        self.round_counter = 0
        self.running = False
        self.federation_thread = threading.Thread(target=self._federation_loop)

    def start(self):
        self.running = True
        self.federation_thread.start()
        print("Federation aggregator started.")

    def stop(self):
        self.running = False
        self.federation_thread.join()
        print("Federation aggregator stopped.")

    def _federation_loop(self):
        while self.running:
            time.sleep(self.federation_interval)
            self.round_counter+=1

            with torch.no_grad():
                all_sd=[c.model.state_dict() for c in self.clients]
                global_sd={}
                for key in all_sd[0].keys():
                    stacked = torch.stack([sd[key].float() for sd in all_sd])
                    if self.round_counter>5:
                        # weighting by f1
                        w=[c.get_performance_weight() for c in self.clients]
                        w=torch.tensor(w).to(stacked.device)
                        dim=stacked.dim()
                        w=w.view(*([len(w)]+[1]*(dim-1)))
                        global_sd[key]=torch.sum(stacked*w,dim=0)
                    else:
                        global_sd[key]=torch.mean(stacked,dim=0)

                # update each client
                for c in self.clients:
                    with c.lock:
                        c.global_model_params=[p.clone() for p in c.model.parameters()]

                        if self.round_counter<10:
                            mix_ratio=0.7
                        else:
                            mix_ratio=min(0.7+(self.round_counter-10)*0.02,0.8)

                        c_sd=c.model.state_dict()
                        for k in global_sd:
                            c_sd[k]= (mix_ratio*global_sd[k] + (1-mix_ratio)*c_sd[k]).detach()

                        try:
                            c.model.load_state_dict(c_sd)
                            c.radius=None
                            print(f"[Aggregator] Round {self.round_counter}: updated client with mix={mix_ratio:.2f}")
                        except RuntimeError as e:
                            print("Error aggregator load:", e)

# -----------------------------------------------------
# 5) Minimal data ingestion: no label usage
# -----------------------------------------------------
def create_data_batches(X,batch_size=800):
    idx=np.arange(len(X))
    while True:
        np.random.shuffle(idx)
        for start in range(0,len(X),batch_size):
            end=min(start+batch_size,len(X))
            yield X[idx[start:end]]

# -----------------------------------------------------
# 6) Simple demonstration: aggregator + 2 clients
# -----------------------------------------------------
def server_mode(
    data_path_1,
    data_path_2,
    run_time=10,
    federation_interval=70,
    seed=42
):
    # 2 clients
    # load data from CSV, but do no label usage
    # just for demonstration, we create random data for each client
    # or you can load real data
    # For demonstration, let's produce random data
    # If you want real data, adapt code here or reference your CSV.
    X1 = np.random.randn(10000,49).astype(np.float32)
    X2 = np.random.randn(12000,49).astype(np.float32)

    # create clients
    c1=OnlineDeepSVDD(input_dim=49,batch_size=800)
    c2=OnlineDeepSVDD(input_dim=49,batch_size=800)
    clients=[c1,c2]

    # aggregator
    aggregator=FederatedServer(clients,federation_interval=federation_interval)
    aggregator.start()
    start_time=time.time()
    gen1=create_data_batches(X1,800)
    gen2=create_data_batches(X2,800)

    while (time.time()-start_time)<(run_time*60):
        try:
            # partial fits
            bx1=next(gen1)
            c1.partial_fit(bx1)
            bx2=next(gen2)
            c2.partial_fit(bx2)
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("Server stopping training loop.")
            break
    aggregator.stop()
    print("Server run complete.")

def client_mode(
    server_address="nwdaf-0.free5gc.org:7000",
    client_id=1
):
    """Here is a placeholder. In reality, you'd connect to aggregator. 
       But we have an internal aggregator in server_mode. 
       If you want real server->client approach, 
       you'd do requests or gRPC to aggregator, etc.
    """
    print(f"Client {client_id} mode not implemented in this minimal code. Sorry!")

# -----------------------------------------------------
# 7) Main
# -----------------------------------------------------
def main():
    parser=argparse.ArgumentParser("Advanced SVDD with aggregator")
    parser.add_argument("--server",action='store_true',help="Run aggregator server mode")
    parser.add_argument("--client",type=str,help="Run aggregator client mode (server_address)")
    parser.add_argument("--data_path_1",type=str,default="data/slice1.csv",help="CSV path #1")
    parser.add_argument("--data_path_2",type=str,default="data/slice2.csv",help="CSV path #2")
    parser.add_argument("--run_time",type=int,default=10,help="Run time in minutes")
    parser.add_argument("--fl_interval",type=int,default=70,help="Federation interval in seconds")
    parser.add_argument("--seed",type=int,default=42,help="Random seed")
    parser.add_argument("--client_id",type=int,default=1,help="Client ID if using client mode")
    args=parser.parse_args()

    set_seed(args.seed)

    if args.server:
        server_mode(
            data_path_1=args.data_path_1,
            data_path_2=args.data_path_2,
            run_time=args.run_time,
            federation_interval=args.fl_interval,
            seed=args.seed
        )
    elif args.client:
        client_mode(
            server_address=args.client,
            client_id=args.client_id
        )
    else:
        parser.print_help()

if __name__=="__main__":
    main()
