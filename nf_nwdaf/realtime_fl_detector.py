import logging
import threading
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from nfstream import NFPlugin, NFStreamer
from scapy.layers.inet import IP, UDP
from scapy.contrib.pfcp import PFCP
import requests
import json
from typing import Dict, List, Optional
import subprocess
import pandas as pd
import os
from time import sleep

def check_docker_containers():
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        container_names = result.stdout.strip().split('\n')
        if container_names and container_names[0]:
            print(f"Docker containers running: {container_names}", flush=True)
            logging.info(f"Docker containers running: {container_names}")
        else:
            print("No Docker containers are running.", flush=True)
            logging.info("No Docker containers are running.")
    except Exception as e:
        print(f"Could not check Docker containers: {e}", flush=True)
        logging.error(f"Could not check Docker containers: {e}")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Docker network configuration
DOCKER_NETWORK_CONFIG = {
    1: {
        'nwdaf_ip': 'nwdaf1',  # Docker service name
        'upf_ip': 'upf1',
        'port': 5000
    },
    2: {
        'nwdaf_ip': 'nwdaf2',
        'upf_ip': 'upf2',
        'port': 5001
    },
    3: {
        'nwdaf_ip': 'nwdaf3',
        'upf_ip': 'upf3',
        'port': 5002
    },
    4: {
        'nwdaf_ip': 'nwdaf4',
        'upf_ip': 'upf4',
        'port': 5003
    }
}

class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(DeepSVDD, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 64)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Ensure correct input dimension
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(-1)}")
        
        # Apply layers with proper shape handling
        x = self.fc1(x)
        if x.size(0) == 1:  # If batch size is 1, repeat the tensor
            x = x.repeat(2, 1)  # Repeat to create a batch size of 2 for BatchNorm
            x = self.bn1(x)
            x = x[0:1]  # Take only the first element back
        else:
            x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) == 1:
            x = x.repeat(2, 1)
            x = self.bn2(x)
            x = x[0:1]
        else:
            x = self.bn2(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        if x.size(0) == 1:
            x = x.repeat(2, 1)
            x = self.bn3(x)
            x = x[0:1]
        else:
            x = self.bn3(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

    def get_parameters(self):
        return [param.detach().clone() for param in self.parameters()]

class OnlineDeepSVDD:
    def __init__(self, input_dim, hidden_dim=384, output_dim=64,
                 batch_size=1024, learning_rate=0.0001, nu=0.8,
                 warmup_batches=100, window_size=500,
                 ema_alpha=0.98, prediction_threshold=1.0,
                 smooth_window=5, weight_decay=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepSVDD(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.is_initialized = False

        # SVDD parameters
        self.center = None
        self.radius = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.nu = nu
        self.anomaly_threshold = None  # Initialize anomaly threshold
        self.initial_threshold = prediction_threshold
        self.adaptive_threshold = True

        # Training state
        self.warmup_batches = warmup_batches
        self.current_batch = 0
        self.window_size = window_size
        self.distance_history = []

        # Smoothing and Prediction
        self.ema_alpha = ema_alpha
        self.ema_center = None
        self.prediction_threshold = prediction_threshold
        self.smooth_window = smooth_window
        self.previous_raw_scores = []

        # Performance tracking
        self.recent_f1_scores = []
        self.performance_window = 5
        self.anomaly_stats = {
            'total_flows': 0,
            'anomaly_count': 0,
            'last_anomaly_time': None
        }

        # Monitoring and Threading
        self.last_update_time = time.time()
        self.lock = threading.Lock()

        # SCAFFOLD components
        self.client_control = None
        self.global_control = None

    def calculate_loss(self, features):
        """Calculate Deep SVDD loss"""
        try:
            if self.center is None:
                # Initialize center as mean of first batch
                self.center = torch.mean(features, dim=0)
                self.anomaly_threshold = self.initial_threshold  # Set initial threshold
                return torch.mean((features - self.center) ** 2)
            
            distances = torch.sum((features - self.center) ** 2, dim=1)
            self.distance_history.extend(distances.detach().cpu().numpy().tolist())
            
            # Keep only recent distances for adaptive threshold
            if len(self.distance_history) > self.window_size:
                self.distance_history = self.distance_history[-self.window_size:]
            
            # Update anomaly threshold adaptively
            if self.adaptive_threshold and len(self.distance_history) >= self.window_size:
                threshold = np.percentile(self.distance_history, (1 - self.nu) * 100)
                self.anomaly_threshold = max(threshold, self.initial_threshold)
            
            return torch.mean(distances)
        except Exception as e:
            logging.error(f"Error in calculate_loss: {str(e)}")
            return None

    def process_flow(self, features):
        """Process a single flow and return anomaly detection result"""
        try:
            # Ensure features is a numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Reshape features to 2D if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                if not self.scaler.n_samples_seen_:
                    self.scaler.partial_fit(features)
                features = self.scaler.transform(features)
            
            features_tensor = torch.FloatTensor(features).to(self.device)
            self.model.eval()
            
            with torch.no_grad():
                output = self.model(features_tensor)
                if self.center is None:
                    self.center = output.mean(dim=0)
                    self.anomaly_threshold = self.initial_threshold
                    return False, 0.0
                
                distance = torch.sum((output - self.center) ** 2).item()
                
                # Update statistics
                self.anomaly_stats['total_flows'] += 1
                is_anomaly = distance > self.anomaly_threshold
                
                if is_anomaly:
                    self.anomaly_stats['anomaly_count'] += 1
                    self.anomaly_stats['last_anomaly_time'] = datetime.now()
                
                return is_anomaly, distance
        except Exception as e:
            logging.error(f"Error in process_flow: {str(e)}")
            return False, 0.0

    def partial_fit(self, X):
        """Partially fit the model on a batch of data"""
        try:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            
            loss = self._perform_training_step(X)
            if loss is not None:
                self.is_initialized = True
            return loss
        except Exception as e:
            logging.error(f"Error in partial_fit: {str(e)}")
            return None

    def _perform_training_step(self, X):
        try:
            # Ensure X is a numpy array
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            
            # Reshape X to 2D if needed
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Fit and transform with scaler
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Convert to tensor and ensure proper shape
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            if X_tensor.dim() == 1:
                X_tensor = X_tensor.unsqueeze(0)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            # SCAFFOLD: Store old parameters
            w_old = None
            if self.client_control is not None:
                w_old = [p.clone().detach() for p in self.model.parameters() if p.requires_grad]
            
            features = self.model(X_tensor)
            loss = self.calculate_loss(features)
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Invalid loss encountered. Skipping update.")
                return None
            
            loss.backward()
            
            # SCAFFOLD: Apply control variates
            if self.client_control is not None and self.global_control is not None:
                grad_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                if len(self.client_control) == len(grad_params) and len(self.global_control) == len(grad_params):
                    for i, param in enumerate(grad_params):
                        c_local = self.client_control[i].to(param.grad.device)
                        c_global = self.global_control[i].to(param.grad.device)
                        param.grad.add_(c_global - c_local)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # SCAFFOLD: Update client control variates
            if self.client_control is not None and self.global_control is not None and w_old is not None:
                w_new = [p.clone().detach() for p in self.model.parameters() if p.requires_grad]
                lr = self.optimizer.param_groups[0]['lr']
                if lr > 1e-9:
                    if len(self.client_control) == len(w_old) and len(self.client_control) == len(w_new) and len(self.client_control) == len(self.global_control):
                        new_client_control = []
                        for i in range(len(self.client_control)):
                            delta_model = w_old[i] - w_new[i]
                            c_local = self.client_control[i].to(self.device)
                            c_global = self.global_control[i].to(self.device)
                            c_update = c_local - c_global + delta_model / lr
                            new_client_control.append(c_update.clone().detach())
                        self.client_control = new_client_control
            
            with torch.no_grad():
                if self.center is not None:
                    if self.ema_center is None:
                        self.ema_center = self.center.clone()
                    else:
                        self.ema_center.mul_(self.ema_alpha).add_(self.center, alpha=1 - self.ema_alpha)
            
            self.current_batch += 1
            self.last_update_time = time.time()
            return loss.item()
        except Exception as e:
            logging.error(f"Error in training step: {str(e)}")
            return None

class FederatedServer:
    def __init__(self, clients, federation_interval=15, test_callback=None, initial_rounds_no_weight=5):
        self.clients = clients
        self.federation_interval = federation_interval
        self.test_callback = test_callback
        self.running = False
        self.federation_thread = None
        self.round_counter = 0
        self.initial_rounds_no_weight = initial_rounds_no_weight
        self.client_performance_history = {i: [] for i in range(len(clients))}
        self.adaptive_mix_ratio = 0.3  # Initial mix ratio

        # Initialize global control variate with momentum
        self.global_control = []
        self.control_momentum = 0.9
        if self.clients:
            first_client = self.clients[0]
            with torch.no_grad():
                self.global_control = [torch.zeros_like(p, device=first_client.device)
                                     for p in first_client.model.model.parameters() if p.requires_grad]
                logging.info(f"Federated Server initialized global control ({len(self.global_control)} tensors).")

    def add_client(self, client_id, client):
        """Add a client to the federation server."""
        self.clients.append(client)
        self.client_performance_history[client_id] = []

    def start(self):
        """Start the federation server."""
        if not self.running:
            self.running = True
            self.federation_thread = threading.Thread(target=self._federation_loop)
            self.federation_thread.daemon = True
            self.federation_thread.start()
            logging.info("Federation server started.")

    def stop(self):
        """Stop the federation server."""
        if self.running:
            self.running = False
            if self.federation_thread:
                self.federation_thread.join()
            logging.info("Federation server stopped.")

    def _update_adaptive_mix_ratio(self):
        """Dynamically adjust mix ratio based on client performance history"""
        if self.round_counter < self.initial_rounds_no_weight:
            return 0.3  # Start with conservative mixing
        
        # Calculate performance trends
        recent_performance = []
        for client_id, history in self.client_performance_history.items():
            if len(history) >= 3:
                recent_avg = np.mean(history[-3:])
                recent_performance.append(recent_avg)
        
        if not recent_performance:
            return 0.3
        
        # Adjust mix ratio based on performance stability
        performance_std = np.std(recent_performance)
        if performance_std < 0.1:  # Stable performance
            self.adaptive_mix_ratio = min(0.5, self.adaptive_mix_ratio + 0.05)
        else:  # Unstable performance
            self.adaptive_mix_ratio = max(0.2, self.adaptive_mix_ratio - 0.05)
        
        return self.adaptive_mix_ratio

    def _update_global_control(self, client_controls):
        """Update global control with momentum"""
        with torch.no_grad():
            for i, global_ctrl in enumerate(self.global_control):
                # Calculate average of client controls
                avg_client_ctrl = torch.mean(torch.stack([ctrl[i] for ctrl in client_controls]), dim=0)
                # Apply momentum update
                global_ctrl.mul_(self.control_momentum).add_(avg_client_ctrl, alpha=1 - self.control_momentum)

    def _federation_loop(self):
        self._initialize_client_controls_if_needed()
        while self.running:
            time.sleep(self.federation_interval)
            if not self.running: break
            self.round_counter += 1
            logging.info(f"\n--- Starting FL Round {self.round_counter} ---")

            # Gather active clients
            active_clients = []
            now = time.time()
            activity_timeout = max(20, self.federation_interval * 2)
            for i, client in enumerate(self.clients):
                if client.is_initialized and (now - client.last_update_time < activity_timeout):
                    if isinstance(client.client_control, list) and len(client.client_control) > 0:
                        active_clients.append(client)
                        # Update performance history
                        if hasattr(client, 'recent_f1_scores') and client.recent_f1_scores:
                            self.client_performance_history[i].append(np.mean(client.recent_f1_scores[-3:]))
                if client.global_control is None or len(client.global_control) != len(self.global_control):
                    client.global_control = self.global_control

            if len(active_clients) < 2:
                logging.info(f"FL Round {self.round_counter}: Not enough active clients ({len(active_clients)}). Skipping aggregation.")
                continue

            # Calculate adaptive mix ratio
            mix_ratio = self._update_adaptive_mix_ratio()
            logging.info(f"FL Round {self.round_counter}: Using adaptive mix ratio {mix_ratio:.3f}")

            # Aggregate models with performance-based weighting
            try:
                with torch.no_grad():
                    all_state_dicts = [client.model.state_dict() for client in active_clients]
                    aggregated_model_state = {}
                    
                    # Calculate performance-based weights
                    client_weights = []
                    for client in active_clients:
                        if hasattr(client, 'recent_f1_scores') and client.recent_f1_scores:
                            weight = np.mean(client.recent_f1_scores[-3:])
                        else:
                            weight = 1.0
                        client_weights.append(weight)
                    
                    # Normalize weights
                    total_weight = sum(client_weights)
                    if total_weight < 1e-8:
                        client_weights = [1.0] * len(active_clients)
                        total_weight = len(active_clients)
                    normalized_weights = [w / total_weight for w in client_weights]
                    
                    # Aggregate parameters
                    first_sd = all_state_dicts[0]
                    for key in first_sd.keys():
                        if first_sd[key].dtype.is_floating_point:
                            try:
                                stacked_params = torch.stack([sd[key].to(first_sd[key].device).float() for sd in all_state_dicts])
                                weights_t = torch.tensor(normalized_weights, dtype=torch.float32, device=stacked_params.device)
                                view_shape = [len(normalized_weights)] + [1] * (stacked_params.dim() - 1)
                                weights_t = weights_t.view(*view_shape)
                                aggregated_model_state[key] = torch.sum(stacked_params * weights_t, dim=0)
                            except RuntimeError as e:
                                logging.error(f"Error stacking/avg param '{key}': {e}. Skipping.")
                                aggregated_model_state[key] = first_sd[key].clone()
                        else:
                            aggregated_model_state[key] = first_sd[key].clone()

                    # Update global control variates
                    client_controls = [client.client_control for client in active_clients]
                    self._update_global_control(client_controls)

                    # Distribute updates to clients
                    for client in self.clients:
                        try:
                            with client.lock:
                                if aggregated_model_state:
                                    client_state = client.model.state_dict()
                                    updated_state = {}
                                    for key in aggregated_model_state:
                                        if key in client_state:
                                            try:
                                                global_param = aggregated_model_state[key].to(client.device)
                                                local_param = client_state[key]
                                                if local_param.dtype != global_param.dtype:
                                                    global_param = global_param.to(local_param.dtype)
                                                updated_state[key] = (1 - mix_ratio) * local_param + mix_ratio * global_param
                                            except Exception as mix_err:
                                                logging.error(f"Error mixing param '{key}': {mix_err}.")
                                                updated_state[key] = client_state[key]
                                        else:
                                            updated_state[key] = aggregated_model_state[key].to(client.device)
                                    client.model.load_state_dict(updated_state)
                                    client.radius = torch.tensor(0.0, device=client.device, dtype=torch.float32)
                                    client.distance_history = []
                        except Exception as e:
                            logging.error(f"Error updating client: {e}")

            except Exception as e:
                logging.error(f"Error during federation round {self.round_counter}: {e}")

        logging.info("Federation loop finished.")

    def _initialize_client_controls_if_needed(self):
        """Initialize client controls if needed."""
        if not hasattr(self, 'global_control') or not self.global_control:
            if self.clients:
                first_client = self.clients[0]
                with torch.no_grad():
                    device = first_client.model.model.fc1.weight.device
                    self.global_control = [torch.zeros_like(p, device=device)
                                         for p in first_client.model.model.parameters() if p.requires_grad]
                    logging.info(f"Initialized global control ({len(self.global_control)} tensors).")

class RealTimeFLAnomalyDetector:
    def __init__(self, slice_id: int, input_dim: int = 10):
        self.slice_id = slice_id
        self.model = OnlineDeepSVDD(input_dim)
        self.flow_buffer = []
        self.batch_size = 1000
        self.last_federation_time = time.time()
        self.federation_interval = 70
        self.config = DOCKER_NETWORK_CONFIG[slice_id]
        self.server = FederatedServer([], federation_interval=15)  # Initialize with empty list
        self.server.add_client(slice_id, self)
        self.server.start()
        
        # Add periodic statistics logging
        self.last_stats_time = time.time()
        self.stats_interval = 300  # Log statistics every 5 minutes
        
        # Initialize federation attributes
        self.is_initialized = True
        self.last_update_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize client control and global control
        self.client_control = []
        self.global_control = None  # Will be set by federation server
        
        # Initialize performance metrics
        self.anomaly_stats = {
            'total_flows': 0,
            'anomaly_count': 0,
            'last_anomaly_time': None
        }

    def process_flow(self, flow_features):
        self.flow_buffer.append(flow_features)
        
        if len(self.flow_buffer) >= self.batch_size:
            batch = np.array(self.flow_buffer[:self.batch_size])
            self.flow_buffer = self.flow_buffer[self.batch_size:]
            
            loss = self.model.partial_fit(batch)
            
            current_time = time.time()
            if current_time - self.last_federation_time >= self.federation_interval:
                self.last_federation_time = current_time
                self._participate_in_federation()
        
        if self.model.anomaly_threshold is not None:
            features = torch.FloatTensor(flow_features).to(self.model.device)
            with torch.no_grad():
                model_features = self.model.model(features)
                distance = torch.sum((model_features - self.model.center) ** 2)
                is_anomaly = distance > self.model.anomaly_threshold
                return is_anomaly, distance.item()
        return False, 0.0

    def _participate_in_federation(self):
        """Participate in federation with other NWDAFs"""
        try:
            # Get current model parameters
            params = self.server.get_model_parameters()
            
            # Convert parameters to JSON-serializable format
            params_dict = {
                'slice_id': self.slice_id,
                'parameters': [p.cpu().numpy().tolist() for p in params]
            }
            
            # Send parameters to other NWDAFs
            for other_slice_id, other_config in DOCKER_NETWORK_CONFIG.items():
                if other_slice_id != self.slice_id:
                    url = f"http://{other_config['nwdaf_ip']}:{other_config['port']}/federation"
                    try:
                        response = requests.post(url, json=params_dict)
                        if response.status_code == 200:
                            other_params = response.json()['parameters']
                            other_params = [torch.tensor(p) for p in other_params]
                            self.server.update_model_parameters(other_params)
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Failed to communicate with NWDAF {other_slice_id}: {e}")
            
            logging.info(f"Slice {self.slice_id}: Participated in federation round")
            
        except Exception as e:
            logging.error(f"Error during federation: {e}")

    def _log_statistics(self):
        """Log periodic statistics about anomaly detection"""
        stats = self.model.anomaly_stats
        if stats['total_flows'] > 0:
            logging.info(
                f"\n{'='*80}\n"
                f"📈 PERIODIC STATISTICS - Slice {self.slice_id}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total flows processed: {stats['total_flows']}\n"
                f"Total anomalies detected: {stats['anomaly_count']}\n"
                f"Overall anomaly rate: {(stats['anomaly_count']/stats['total_flows']*100):.2f}%\n"
                f"Last anomaly detected: {stats['last_anomaly_time']}\n"
                f"Current threshold: {self.model.anomaly_threshold:.4f if self.model.anomaly_threshold is not None else 'N/A'}\n"
                f"{'='*80}\n"
            )

class PFCPFlowGenerator(NFPlugin):
    def __init__(self, slice_id):
        super().__init__()
        self.slice_id = slice_id
        self.detector = RealTimeFLAnomalyDetector(slice_id)
        self.flow_features = []
        self.last_packet_time = {}  # Track last packet time for each flow
        self.packet_times = {}  # Track packet arrival times for each flow

    def _calculate_features(self, flow):
        """Calculate comprehensive flow features similar to FL4.py"""
        # Basic flow features
        features = [
            flow.src_port,
            flow.dst_port,
            flow.bidirectional_duration_ms,
            flow.bidirectional_packets,
            flow.bidirectional_bytes,
            flow.src2dst_duration_ms,
            flow.src2dst_packets,
            flow.src2dst_bytes,
            flow.dst2src_duration_ms,
            flow.dst2src_packets,
            flow.dst2src_bytes,
            flow.bidirectional_min_ps,
            flow.bidirectional_mean_ps,
            flow.bidirectional_stddev_ps,
            flow.bidirectional_max_ps,
            flow.src2dst_min_ps,
            flow.src2dst_mean_ps,
            flow.src2dst_stddev_ps,
            flow.src2dst_max_ps,
            flow.dst2src_min_ps,
            flow.dst2src_mean_ps,
            flow.dst2src_stddev_ps,
            flow.dst2src_max_ps,
            flow.bidirectional_min_piat_ms,
            flow.bidirectional_mean_piat_ms,
            flow.bidirectional_stddev_piat_ms,
            flow.bidirectional_max_piat_ms,
            flow.src2dst_min_piat_ms,
            flow.src2dst_mean_piat_ms,
            flow.src2dst_stddev_piat_ms,
            flow.src2dst_max_piat_ms,
            flow.dst2src_min_piat_ms,
            flow.dst2src_mean_piat_ms,
            flow.dst2src_stddev_piat_ms,
            flow.dst2src_max_piat_ms,
            flow.bidirectional_syn_packets,
            flow.bidirectional_cwr_packets,
            flow.bidirectional_ece_packets,
            flow.bidirectional_urg_packets,
            flow.bidirectional_ack_packets,
            flow.bidirectional_psh_packets,
            flow.bidirectional_rst_packets,
            flow.bidirectional_fin_packets,
            flow.src2dst_syn_packets,
            flow.src2dst_cwr_packets,
            flow.src2dst_ece_packets,
            flow.src2dst_urg_packets,
            flow.src2dst_ack_packets,
            flow.src2dst_psh_packets,
            flow.src2dst_rst_packets,
            flow.src2dst_fin_packets,
            flow.dst2src_syn_packets,
            flow.dst2src_cwr_packets,
            flow.dst2src_ece_packets,
            flow.dst2src_urg_packets,
            flow.dst2src_ack_packets,
            flow.dst2src_psh_packets,
            flow.dst2src_rst_packets,
            flow.dst2src_fin_packets
        ]
        return features

    def _is_anomaly(self, flow, distance, threshold):
        """Determine if the flow is anomalous based on various characteristics"""
        # High packet count with small packet size (DDoS)
        if flow.bidirectional_packets > 1000 and flow.bidirectional_mean_ps < 100:
            return True
            
        # High packet count with short duration (Brute Force)
        if flow.bidirectional_packets > 50 and flow.bidirectional_duration_ms < 1000:
            return True
            
        # Unusual TCP flag combinations
        if (flow.bidirectional_syn_packets > 0 and flow.bidirectional_fin_packets == 0 and 
            flow.bidirectional_duration_ms > 5000):
            return True
            
        # High distance from normal behavior
        if distance > threshold * 1.5:
            return True
            
        return False

    def on_update(self, packet, flow):
        # Calculate comprehensive features
        features = self._calculate_features(flow)
        
        is_anomaly, distance = self.detector.process_flow(features)
        
        # Check for additional anomaly conditions
        if not is_anomaly and self._is_anomaly(flow, distance, self.detector.model.anomaly_threshold):
            is_anomaly = True
        
        # Real-time logging for each flow
        timestamp = datetime.now().strftime('%H:%M:%S')
        status = "🚨 ANOMALY" if is_anomaly else "✅ BENIGN"
        protocol = "TCP" if flow.protocol == 6 else "UDP" if flow.protocol == 17 else "OTHER"
        
        logging.info(
            f"{timestamp} | {status} | Slice {self.slice_id} | "
            f"Flow: {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | "
            f"Proto: {protocol} | "
            f"Pkts: {flow.bidirectional_packets} | Bytes: {flow.bidirectional_bytes} | "
            f"Duration: {flow.bidirectional_duration_ms}ms | "
            f"Dist: {distance:.4f}"
        )

        # Additional warning for anomalies
        if is_anomaly:
            logging.warning(
                f"\n{'='*80}\n"
                f"⚠️ ANOMALY DETECTED - Slice {self.slice_id}\n"
                f"Time: {timestamp}\n"
                f"Distance: {distance:.4f} (Threshold: {self.detector.model.anomaly_threshold:.4f})\n"
                f"Flow Details:\n"
                f"  Source IP: {flow.src_ip}\n"
                f"  Destination IP: {flow.dst_ip}\n"
                f"  Protocol: {protocol}\n"
                f"  Packets: {flow.bidirectional_packets}\n"
                f"  Bytes: {flow.bidirectional_bytes}\n"
                f"  Duration: {flow.bidirectional_duration_ms}ms\n"
                f"  Mean Packet Size: {flow.bidirectional_mean_ps} bytes\n"
                f"{'='*80}\n"
            )

# Define the mapping of slices to CSV files
SLICE_FILE_MAPPING = {
    1: ["output_benign_label.csv", "output_ddos2_label.csv"],  # Slice 1: Mix of benign and DDoS
    2: ["output_benign_label.csv", "output_bf_label.csv"],     # Slice 2: Mix of benign and brute force
    3: ["output_benign_label.csv", "output_pfcp0_label.csv"],  # Slice 3: Mix of benign and PFCP
    4: ["output_benign_label.csv"]                             # Slice 4: Only benign traffic
}

def load_csv_data(slice_id):
    """Load and process CSV data for a specific slice"""
    dfs = []
    base_path = "/app/flow_data"
    
    if slice_id not in SLICE_FILE_MAPPING:
        logging.error(f"No file mapping found for slice {slice_id}")
        return None
        
    for filename in SLICE_FILE_MAPPING[slice_id]:
        file_path = os.path.join(base_path, filename)
        logging.info(f"Attempting to load {file_path}")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Add a source column to track the traffic type
                df['traffic_type'] = filename.split('_')[1]  # Extract type from filename
                dfs.append(df)
                logging.info(f"Successfully loaded {filename} for slice {slice_id}")
            except Exception as e:
                logging.error(f"Error loading {filename} for slice {slice_id}: {e}")
        else:
            logging.error(f"File not found: {file_path}")
    
    if not dfs:
        logging.error(f"No CSV files could be loaded for slice {slice_id}")
        return None
    
    # Combine all dataframes for this slice
    combined_df = pd.concat(dfs, ignore_index=True)
    # Shuffle the combined data to mix different traffic types
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    logging.info(f"Total records for slice {slice_id}: {len(combined_df)}")
    return combined_df

def process_csv_data(slice_id, df, detector):
    """Process CSV data for a given slice with batch processing"""
    if df is None or len(df) == 0:
        logging.warning(f"No data to process for slice {slice_id}")
        return

    batch_size = 32  # Process data in batches for efficiency
    total_processed = 0
    anomalies_detected = 0
    
    try:
        # Process data in batches
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            for _, row in batch_df.iterrows():
                try:
                    # Extract features from the CSV row
                    features = [
                        float(row.get('src_port', 0)),
                        float(row.get('dst_port', 0)),
                        float(row.get('duration', 0)),
                        float(row.get('total_pkts', 0)),
                        float(row.get('total_bytes', 0)),
                        float(row.get('src2dst_bytes', 0)),
                        float(row.get('dst2src_bytes', 0)),
                        float(row.get('protocol', 0)),
                        float(row.get('pkt_size_avg', 0)),
                        float(row.get('pkt_size_std', 0))
                    ]
                    
                    is_anomaly, distance = detector.process_flow(features)
                    total_processed += 1
                    
                    if is_anomaly:
                        anomalies_detected += 1
                        # Log anomaly with detailed information
                        logging.warning(
                            f"\n{'='*80}\n"
                            f"⚠️ ANOMALY DETECTED - Slice {slice_id}\n"
                            f"Traffic Type: {row.get('traffic_type', 'unknown')}\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Flow: {row.get('src_ip', 'unknown')}:{row.get('src_port', 0)} -> "
                            f"{row.get('dst_ip', 'unknown')}:{row.get('dst_port', 0)}\n"
                            f"Distance: {distance:.4f}\n"
                            f"Threshold: {detector.model.anomaly_threshold:.4f}\n"
                            f"Protocol: {int(row.get('protocol', 0))}\n"
                            f"Total Packets: {int(row.get('total_pkts', 0))}\n"
                            f"Total Bytes: {int(row.get('total_bytes', 0))}\n"
                            f"{'='*80}\n"
                        )
                    
                    # Log progress every 1000 records
                    if total_processed % 1000 == 0:
                        logging.info(
                            f"Slice {slice_id} Progress: {total_processed}/{len(df)} records processed. "
                            f"Anomalies detected: {anomalies_detected}"
                        )
                
                except Exception as e:
                    logging.error(f"Error processing record in slice {slice_id}: {str(e)}")
                    continue
            
            # Optional: Add a small delay between batches to prevent overwhelming the system
            time.sleep(0.1)
        
        # Log final statistics
        logging.info(
            f"\nProcessing completed for slice {slice_id}:\n"
            f"Total records processed: {total_processed}\n"
            f"Anomalies detected: {anomalies_detected}\n"
            f"Anomaly rate: {(anomalies_detected/total_processed)*100:.2f}%\n"
        )
        
    except Exception as e:
        logging.error(f"Fatal error processing CSV data for slice {slice_id}: {str(e)}")

def start_realtime_detection(iface='eth0', max_nflows=100000):
    streamers = {}
    last_flow_time = {}
    lock = threading.Lock()

    def monitor_no_data():
        """Monitor for periods of no data reception and handle CSV data processing"""
        logging.info("Starting monitor thread for CSV data processing")
        detectors = {}  # Cache detectors for each slice
        processed_files = set()  # Track processed files to avoid duplicates
        
        while True:
            time.sleep(30)
            now = time.time()
            with lock:
                for slice_id in range(1, 5):
                    last = last_flow_time.get(slice_id, 0)
                    if last == 0 or now - last > 30:
                        msg = f"No network flows for slice {slice_id} in the last 30 seconds, processing CSV data..."
                        print(msg, flush=True)
                        logging.info(msg)
                        
                        # Get or create detector for this slice
                        if slice_id not in detectors:
                            detectors[slice_id] = RealTimeFLAnomalyDetector(slice_id)
                            logging.info(f"Created new detector for slice {slice_id}")
                        
                        # Try to load CSV data
                        df = load_csv_data(slice_id)
                        if df is not None:
                            # Create a unique identifier for this slice's data
                            data_id = f"slice_{slice_id}_{len(df)}"
                            if data_id not in processed_files:
                                logging.info(f"Processing {len(df)} records for slice {slice_id}")
                                process_csv_data(slice_id, df, detectors[slice_id])
                                processed_files.add(data_id)
                                logging.info(f"Completed processing CSV data for slice {slice_id}")
                            else:
                                logging.info(f"Data for slice {slice_id} already processed, skipping")
                        else:
                            logging.warning(f"No CSV data available for slice {slice_id}")
                    else:
                        logging.info(f"Network flows active for slice {slice_id}, skipping CSV processing")

    def run_streamer(slice_id, streamer):
        nonlocal last_flow_time
        flow_count = 0
        csv_processed = False
        
        try:
            for flow in streamer:
                with lock:
                    last_flow_time[slice_id] = time.time()
                    flow_count += 1
                    
                if flow_count >= max_nflows:
                    break
        except Exception as e:
            logging.error(f"Error processing network flows for slice {slice_id}: {e}")
        
        # If no network flows or error, try CSV data
        if flow_count == 0 and not csv_processed:
            logging.info(f"No network flows detected for slice {slice_id}, trying CSV data...")
            df = load_csv_data(slice_id)
            if df is not None:
                detector = RealTimeFLAnomalyDetector(slice_id)
                process_csv_data(slice_id, df, detector)
                csv_processed = True
            else:
                logging.warning(f"No CSV data available for slice {slice_id}")

    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_no_data)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Start NFStreamer for each slice
    for slice_id in range(1, 5):
        try:
            logging.info(f"Started NFStreamer for slice {slice_id}")
            streamer = NFStreamer(source=iface, decode_tunnels=True)
            streamers[slice_id] = streamer
        except Exception as e:
            logging.error(f"Error creating NFStreamer for slice {slice_id}: {e}")
            continue

    # Start streaming threads
    threads = []
    for slice_id, streamer in streamers.items():
        try:
            thread = threading.Thread(target=run_streamer, args=(slice_id, streamer))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            logging.info(f"Started streaming thread for slice {slice_id}")
        except Exception as e:
            logging.error(f"Error starting thread for slice {slice_id}: {e}")
            continue

    if not threads:
        logging.error("No streaming threads were successfully started. Exiting.")
        return

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    import argparse
    import os
    import requests
    from time import sleep

    parser = argparse.ArgumentParser()
    parser.add_argument('--slice_id', type=int, required=True, help='Slice ID')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file to process')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    args = parser.parse_args()

    # Get aggregator URL from environment
    AGGREGATOR_URL = os.getenv('AGGREGATOR_URL', 'http://nwdaf-aggregator:5000')

    # Register with aggregator
    try:
        response = requests.post(f"{AGGREGATOR_URL}/register", json={"slice_id": args.slice_id})
        if response.status_code != 200:
            logging.error(f"Failed to register with aggregator: {response.text}")
            exit(1)
        logging.info(f"Successfully registered with aggregator for slice {args.slice_id}")
    except Exception as e:
        logging.error(f"Error registering with aggregator: {str(e)}")
        exit(1)

    # Initialize detector
    detector = RealTimeFLAnomalyDetector(args.slice_id)
    
    def process_csv_continuously():
        while True:
            try:
                # Load CSV data
                df = load_csv_data(args.slice_id)
                if df is None:
                    logging.error(f"Could not load CSV file for slice {args.slice_id}")
                    sleep(10)
                    continue

                # Process data in batches
                total_processed = 0
                anomalies_detected = 0
                
                while total_processed < len(df):
                    # Get next batch
                    end_idx = min(total_processed + args.batch_size, len(df))
                    batch_df = df.iloc[total_processed:end_idx]
                    
                    # Process batch
                    for _, row in batch_df.iterrows():
                        try:
                            features = [
                                float(row.get('src_port', 0)),
                                float(row.get('dst_port', 0)),
                                float(row.get('duration', 0)),
                                float(row.get('total_pkts', 0)),
                                float(row.get('total_bytes', 0)),
                                float(row.get('src2dst_bytes', 0)),
                                float(row.get('dst2src_bytes', 0)),
                                float(row.get('protocol', 0)),
                                float(row.get('pkt_size_avg', 0)),
                                float(row.get('pkt_size_std', 0))
                            ]
                            
                            is_anomaly, distance = detector.process_flow(features)
                            total_processed += 1
                            
                            if is_anomaly:
                                anomalies_detected += 1
                                logging.warning(
                                    f"\n{'='*80}\n"
                                    f"⚠️ ANOMALY DETECTED - Slice {args.slice_id}\n"
                                    f"Traffic Type: {row.get('traffic_type', 'unknown')}\n"
                                    f"Flow: {row.get('src_ip', 'unknown')}:{row.get('src_port', 0)} -> "
                                    f"{row.get('dst_ip', 'unknown')}:{row.get('dst_port', 0)}\n"
                                    f"Distance: {distance:.4f}\n"
                                    f"Protocol: {int(row.get('protocol', 0))}\n"
                                    f"{'='*80}\n"
                                )
                            
                            # Log progress every 100 records
                            if total_processed % 100 == 0:
                                logging.info(
                                    f"Slice {args.slice_id} Progress: {total_processed}/{len(df)} "
                                    f"({(total_processed/len(df)*100):.1f}%) - "
                                    f"Anomalies: {anomalies_detected}"
                                )
                                
                                # Send update to aggregator
                                try:
                                    model_params = [param.cpu().detach().numpy().tolist() 
                                                  for param in detector.model.model.parameters()]
                                    metrics = {
                                        'anomaly_rate': anomalies_detected / total_processed,
                                        'total_processed': total_processed
                                    }
                                    
                                    response = requests.post(
                                        f"{AGGREGATOR_URL}/update",
                                        json={
                                            "slice_id": args.slice_id,
                                            "model_params": model_params,
                                            "metrics": metrics
                                        }
                                    )
                                    
                                    if response.status_code == 200:
                                        # Update local model with global model
                                        global_model = response.json().get('global_model')
                                        if global_model:
                                            for param, global_param in zip(
                                                detector.model.model.parameters(),
                                                [torch.tensor(p) for p in global_model]
                                            ):
                                                param.data.copy_(global_param)
                                            logging.info("Updated local model with global parameters")
                                    
                                except Exception as e:
                                    logging.error(f"Error updating aggregator: {str(e)}")
                        
                        except Exception as e:
                            logging.error(f"Error processing record: {str(e)}")
                            continue
                    
                    # Add a small delay between batches
                    sleep(0.1)
                
                logging.info(
                    f"\nCompleted processing cycle for slice {args.slice_id}:\n"
                    f"Total records processed: {total_processed}\n"
                    f"Anomalies detected: {anomalies_detected}\n"
                    f"Anomaly rate: {(anomalies_detected/total_processed)*100:.2f}%\n"
                )
                
                # Small delay before starting next cycle
                sleep(1)
                
            except Exception as e:
                logging.error(f"Error in processing cycle: {str(e)}")
                sleep(5)
                continue

    # Start continuous processing
    process_csv_continuously() 