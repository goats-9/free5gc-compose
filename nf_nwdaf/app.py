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
from flask import Flask, jsonify
from torch.nn import functional as F

# --------------- Additional NFStreamer imports for real-time ---------------
try:
    from nfstream import NFPlugin, NFStreamer
    from scapy.layers.inet import IP, UDP
    from scapy.contrib.pfcp import PFCP, PFCPmessageType
    NFSTREAM_AVAILABLE = True
except ImportError:
    # If NFStreamer isn't installed in this environment, we can't do real-time flows
    NFSTREAM_AVAILABLE = False

###############################################################################
#                     1) Utility for reproducibility
###############################################################################
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################################
#                     2) DeepSVDD network
###############################################################################
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(DeepSVDD, self).__init__()
        # Improved architecture with residual connections
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 64)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

    def get_parameters(self):
        return [param.detach().clone() for param in self.parameters()]

###############################################################################
#                     3) Online Deep SVDD Client
###############################################################################
class OnlineDeepSVDD:
    def __init__(self, input_dim, batch_size=1000, learning_rate=0.002):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepSVDD(input_dim).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-7,  # Adjusted weight decay
            amsgrad=True       # Enable AMSGrad variant
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Improved parameters
        self.nu = 0.3  # Reduced anomaly ratio
        self.warmup_batches = 800
        self.window_size = 4000
        self.quantile_threshold = 0.95
        
        # Initialize other attributes
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.center = None
        self.radius = None
        self.distance_history = []
        self.global_model_params = None
        self.lock = threading.Lock()
        self.detection_history = []
        self.anomaly_threshold = None

    def partial_fit(self, X_batch, y_batch=None):
        with self.lock:
            # -----------------------------------------------------
            # 1. Record inference start time
            # -----------------------------------------------------
            inference_time_start = datetime.now()

            X_batch = torch.FloatTensor(X_batch).to(self.device)
            self.model.train()
            
            # Forward pass
            features = self.model(X_batch)
            
            # Center calculation with momentum
            if self.center is None:
                self.center = torch.mean(features.detach(), dim=0)
            else:
                new_center = torch.mean(features.detach(), dim=0)
                self.center = 0.95 * self.center + 0.05 * new_center
            
            # Distance calculation
            distances = torch.sum((features - self.center) ** 2, dim=1)
            
            # Radius estimation with quantile
            if self.radius is None:
                self.radius = torch.quantile(distances.detach(), self.quantile_threshold)
            else:
                new_radius = torch.quantile(distances.detach(), self.quantile_threshold)
                self.radius = 0.95 * self.radius + 0.05 * new_radius
            
            # Loss function
            loss = torch.mean(distances) + 0.1 * torch.relu(self.radius - torch.mean(distances))
            
            # Add regularization if global model exists
            if self.global_model_params is not None:
                proximal_term = 0
                for w, w_t in zip(self.model.parameters(), self.global_model_params):
                    proximal_term += (w - w_t.to(self.device)).norm(2)
                loss += 0.01 * proximal_term
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step(loss)
            
            # Update distance history
            current_distances = distances.detach().cpu().numpy()
            self.distance_history.extend(current_distances.tolist())
            if len(self.distance_history) > self.window_size:
                self.distance_history = self.distance_history[-self.window_size:]
            
            # Adaptive threshold calculation
            if len(self.distance_history) >= self.window_size:
                self.anomaly_threshold = np.percentile(self.distance_history, 95)
            
            # -----------------------------------------------------
            # 2. Record inference end time and compute duration in ms
            # -----------------------------------------------------
            inference_time_end = datetime.now()
            inference_duration_ms = (inference_time_end - inference_time_start).total_seconds() * 1000
            
            # Logging each data point
            anomaly_count = 0
            for i, dist_val in enumerate(current_distances):
                # Determine if it's an anomaly
                if self.anomaly_threshold is not None:
                    is_anomaly = dist_val > self.anomaly_threshold
                else:
                    is_anomaly = False

                status_str = 'ANOMALY' if is_anomaly else 'NORMAL'
                if is_anomaly:
                    anomaly_count += 1

                logging.info(
                    "[{}] Sample {}: Distance={:.4f}, Status={}, "
                    "Inference took: {:.3f} ms".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        i,
                        dist_val,
                        status_str,
                        inference_duration_ms
                    )
                )
            
            if anomaly_count > 0:
                logging.warning(
                    "[{}] Detected {} anomalies in current batch".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        anomaly_count
                    )
                )
            
            return loss.item()

###############################################################################
#                     4) Federated Server
###############################################################################
class FederatedServer:
    def __init__(self, clients, federation_interval=70):
        self.clients = clients
        self.federation_interval = federation_interval
        self.stop_flag = False
        self.thread = None

    def federated_averaging(self):
        with torch.no_grad():
            # Get parameters from all clients
            all_parameters = [client.model.get_parameters() for client in self.clients]
            
            # Average the parameters
            avg_parameters = []
            for param_set in zip(*all_parameters):
                avg_param = sum(param_set) / len(param_set)
                avg_parameters.append(avg_param)
            
            # Update all clients with the averaged parameters
            for client in self.clients:
                for param, avg_param in zip(client.model.parameters(), avg_parameters):
                    param.data.copy_(avg_param)
                client.global_model_params = [p.clone() for p in avg_parameters]

    def federation_loop(self):
        while not self.stop_flag:
            time.sleep(self.federation_interval)
            self.federated_averaging()

    def start(self):
        self.thread = threading.Thread(target=self.federation_loop)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join()

###############################################################################
#                     5) Minimal data ingestion: no label usage
###############################################################################
def create_data_batches(X, batch_size=800):
    idx = np.arange(len(X))
    while True:
        np.random.shuffle(idx)
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield X[idx[start:end]]

def create_data_batches_with_labels(X, labels, batch_size=800):
    idx = np.arange(len(X))
    while True:
        np.random.shuffle(idx)
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield X[idx[start:end]], labels[idx[start:end]]

###############################################################################
#                     6) Simple demonstration: aggregator + 2 clients
###############################################################################
app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

anomaly_stats = {
    'total_samples': 0,
    'anomalies': 0,
    'normal': 0,
    'current_threshold': None
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/stats')
def get_stats():
    return jsonify(anomaly_stats)

@app.route('/')
def dashboard():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NWDAF Anomaly Detection</title>
        <script>
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total').textContent = data.total_samples;
                        document.getElementById('anomalies').textContent = data.anomalies;
                        document.getElementById('normal').textContent = data.normal;
                        document.getElementById('threshold').textContent = 
                            data.current_threshold ? data.current_threshold.toFixed(4) : 'N/A';
                    });
            }
            setInterval(updateStats, 1000);
        </script>
    </head>
    <body>
        <h1>NWDAF Anomaly Detection Dashboard</h1>
        <p>Total Samples: <span id="total">0</span></p>
        <p>Anomalies: <span id="anomalies">0</span></p>
        <p>Normal: <span id="normal">0</span></p>
        <p>Current Threshold: <span id="threshold">N/A</span></p>
    </body>
    </html>
    '''

def server_mode(
    data_path_1,
    data_path_2,
    run_time=10,
    federation_interval=20,
    seed=42
):
    # Start Flask app in a separate thread
    def run_flask():
        app.run(host='0.0.0.0', port=5000)
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Load data from CSV files
    df1 = pd.read_csv(data_path_1)
    df2 = pd.read_csv(data_path_2)
    
    # Save labels if they exist
    labels1 = None
    labels2 = None
    if 'Label' in df1.columns:
        labels1 = df1['Label'].values
        df1 = df1.drop('Label', axis=1)
    if 'Label' in df2.columns:
        labels2 = df2['Label'].values
        df2 = df2.drop('Label', axis=1)
    
    # Normalize features
    scaler = StandardScaler()
    X1 = scaler.fit_transform(df1.values).astype(np.float32)
    X2 = scaler.transform(df2.values).astype(np.float32)
    
    # Create data generators with labels (if present)
    if labels1 is not None:
        gen1 = create_data_batches_with_labels(X1, labels1, 800)
    else:
        gen1 = create_data_batches(X1, 800)
    
    if labels2 is not None:
        gen2 = create_data_batches_with_labels(X2, labels2, 800)
    else:
        gen2 = create_data_batches(X2, 800)
    
    input_dim = X1.shape[1]
    
    # Create clients
    c1 = OnlineDeepSVDD(input_dim=input_dim, batch_size=800)
    c2 = OnlineDeepSVDD(input_dim=input_dim, batch_size=800)
    clients = [c1, c2]
    
    # Create aggregator
    aggregator = FederatedServer(clients, federation_interval=federation_interval)
    aggregator.start()
    
    start_time = time.time()
    
    print("Starting training loop...")
    while (time.time() - start_time) < (run_time * 60):
        try:
            bx1, _ = next(gen1)
            bx2, _ = next(gen2)
            c1.partial_fit(bx1)
            c2.partial_fit(bx2)
            
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("Server stopping training loop.")
            break
    
    aggregator.stop()
    print("Server run complete.")

###############################################################################
#               (New) Real-Time Mode with NFStreamer (PFCPFlowGenerator)
###############################################################################
class PFCPFlowGenerator(NFPlugin):
    def on_init(self, packet, flow):
        # Initialize counters for each PFCP message type
        for message_type in PFCPmessageType.values():
            setattr(flow.udps, f"{message_type}_counter", 0)

    def on_update(self, packet, flow):
        if packet.protocol != 17 or packet.src_port != 8805 or packet.dst_port != 8805:
            return
        ip_packet = IP(packet.ip_packet)
        try:
            udp_dgram = ip_packet[UDP]
            payload = udp_dgram[PFCP]
        except:
            return
        if not payload:
            return
        try:
            message_type = payload.message_type
            for key, value in PFCPmessageType.items():
                if message_type == key:
                    counter_name = f"{value}_counter"
                    setattr(flow.udps, counter_name, getattr(flow.udps, counter_name) + 1)
                    break
        except IndexError:
            return

def realtime_mode(
    iface='eth0',
    batch_size=20,
    max_nflows=100000,
    seed=42
):
    """
    Capture PFCP flows in real-time using NFStreamer + PFCPFlowGenerator,
    and feed them to the OnlineDeepSVDD in batches.
    """
    if not NFSTREAM_AVAILABLE:
        logging.error("NFStreamer or scapy not installed, cannot run realtime mode.")
        return

    set_seed(seed)

    logging.info(f"Starting NFStreamer on iface={iface}, max_nflows={max_nflows}, batch_size={batch_size}")

    # Start NFStreamer
    streamer = NFStreamer(
        source=iface,
        active_timeout=10,
        idle_timeout=1,
        max_nflows=max_nflows,
        udps=PFCPFlowGenerator(),
        statistical_analysis=True
    )

    # We won't know our input_dim until we see the first flow
    # because we want "all features" from the flow dictionary.
    input_dim = None
    model = None

    flow_buffer = []

    for flow in streamer:
        flow_dict = dict(zip(flow.keys(), flow.values()))
        # Determine our feature keys from the first flow
        if input_dim is None:
            feature_keys = sorted(flow_dict.keys())
            input_dim = len(feature_keys)
            model = OnlineDeepSVDD(input_dim=input_dim, batch_size=batch_size)
            logging.info(f"Discovered {input_dim} flow features. Using all for Deep SVDD.")
        
        # Convert flow_dict to numeric array
        feat_vector = []
        for k in sorted(flow_dict.keys()):
            val = flow_dict[k]
            try:
                fv = float(val)
            except:
                fv = 0.0
            feat_vector.append(fv)
        feat_vector = np.array(feat_vector, dtype=np.float32)

        flow_buffer.append(feat_vector)

        # If we hit batch_size, partial_fit, then clear buffer
        if len(flow_buffer) >= batch_size:
            batch_arr = np.stack(flow_buffer, axis=0)
            model.partial_fit(batch_arr)
            flow_buffer.clear()

    # If leftover flows in buffer
    if flow_buffer:
        batch_arr = np.stack(flow_buffer, axis=0)
        model.partial_fit(batch_arr)
        flow_buffer.clear()

    logging.info("Real-time streaming completed (end of NFStreamer).")

###############################################################################
#                                Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='Real-Time Federated Learning with Deep SVDD')
    parser.add_argument('--server', action='store_true', help='Run as server')
    parser.add_argument('--client', type=str, help='Run as client and specify server address')
    parser.add_argument('--data_path_1', type=str, help='Path to first data CSV file')
    parser.add_argument('--data_path_2', type=str, help='Path to second data CSV file')
    parser.add_argument('--run_time', type=int, default=2, help='Run time in minutes')
    parser.add_argument('--fl_interval', type=int, default=20, help='Federation interval in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # -------------------------- New Arg for Realtime --------------------------
    parser.add_argument('--realtime', action='store_true',
                        help='Run NFStreamer in real-time to capture PFCP flows.')

    parser.add_argument('--iface', type=str, default='eth0',
                        help='Interface to capture from in realtime mode.')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for real-time partial_fit.')
    parser.add_argument('--max_nflows', type=int, default=100000,
                        help='Max flows to track in realtime mode.')

    args = parser.parse_args()

    if args.realtime:
        logging.info("Starting real-time mode...")
        realtime_mode(
            iface=args.iface,
            batch_size=args.batch_size,
            max_nflows=args.max_nflows,
            seed=args.seed
        )
    elif args.server:
        print("Starting server mode...")
        server_mode(
            data_path_1=args.data_path_1,
            data_path_2=args.data_path_2,
            run_time=args.run_time,
            federation_interval=args.fl_interval,
            seed=args.seed
        )
    else:
        print("Client mode not implemented yet")
        sys.exit(1)

if __name__ == "__main__":
    main()
\begin{figure}[!ht]
    \centering
    \includegraphics[width=\linewidth]{slice-1-f1-comp.pdf}
    \caption{Cross-slice federated learning vs Single-Slice}
    \label{fig:results3}
\end{figure}
