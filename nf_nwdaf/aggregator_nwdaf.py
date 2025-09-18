import logging
import threading
import time
from datetime import datetime
import torch
import numpy as np
from flask import Flask, request, jsonify
from realtime_fl_detector import OnlineDeepSVDD, DeepSVDD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class AggregatorNWDAF:
    def __init__(self, input_dim=10):
        self.input_dim = input_dim
        self.global_model = OnlineDeepSVDD(input_dim)
        self.client_models = {}
        self.client_weights = {}
        self.aggregation_lock = threading.Lock()
        self.last_aggregation = time.time()
        self.aggregation_interval = 30  # Aggregate every 30 seconds
        
    def register_client(self, slice_id):
        """Register a new client/slice"""
        with self.aggregation_lock:
            if slice_id not in self.client_models:
                self.client_models[slice_id] = []
                self.client_weights[slice_id] = 1.0
                logger.info(f"Registered new client for slice {slice_id}")
    
    def update_client_model(self, slice_id, model_params, metrics):
        """Receive and store client model updates"""
        with self.aggregation_lock:
            self.register_client(slice_id)
            self.client_models[slice_id] = model_params
            
            # Update client weights based on performance metrics
            if metrics and 'anomaly_rate' in metrics:
                # Adjust weight based on anomaly detection rate
                self.client_weights[slice_id] = max(0.1, min(1.0, 1.0 - metrics['anomaly_rate']))
            
            logger.info(f"Received model update from slice {slice_id}")
            
            # Check if it's time to aggregate
            if time.time() - self.last_aggregation >= self.aggregation_interval:
                self._aggregate_models()
    
    def _aggregate_models(self):
        """Aggregate models from all clients using weighted averaging"""
        try:
            if not self.client_models:
                return
            
            logger.info("Starting model aggregation...")
            
            # Normalize weights
            total_weight = sum(self.client_weights.values())
            normalized_weights = {k: w/total_weight for k, w in self.client_weights.items()}
            
            # Initialize aggregated parameters
            aggregated_params = []
            first_client = list(self.client_models.values())[0]
            
            for param_idx in range(len(first_client)):
                weighted_param = torch.zeros_like(first_client[param_idx])
                for slice_id, client_params in self.client_models.items():
                    weight = normalized_weights[slice_id]
                    weighted_param += client_params[param_idx] * weight
                aggregated_params.append(weighted_param)
            
            # Update global model
            self.global_model.model.load_state_dict({
                name: param for name, param in zip(
                    self.global_model.model.state_dict().keys(),
                    aggregated_params
                )
            })
            
            self.last_aggregation = time.time()
            logger.info("Model aggregation completed successfully")
            
            # Log aggregation statistics
            logger.info(f"Aggregation stats:")
            for slice_id, weight in normalized_weights.items():
                logger.info(f"Slice {slice_id}: weight = {weight:.3f}")
            
        except Exception as e:
            logger.error(f"Error during model aggregation: {str(e)}")
    
    def get_global_model(self):
        """Return the current global model parameters"""
        return [param.clone().detach() for param in self.global_model.model.parameters()]

# Flask application for REST API
app = Flask(__name__)
aggregator = AggregatorNWDAF()

@app.route('/register', methods=['POST'])
def register_client():
    data = request.json
    slice_id = data.get('slice_id')
    if slice_id is not None:
        aggregator.register_client(slice_id)
        return jsonify({"status": "success", "message": f"Registered slice {slice_id}"})
    return jsonify({"status": "error", "message": "Missing slice_id"}), 400

@app.route('/update', methods=['POST'])
def update_model():
    data = request.json
    slice_id = data.get('slice_id')
    model_params = data.get('model_params')
    metrics = data.get('metrics', {})
    
    if slice_id is not None and model_params is not None:
        # Convert model parameters from JSON to tensors
        model_params = [torch.tensor(param) for param in model_params]
        aggregator.update_client_model(slice_id, model_params, metrics)
        return jsonify({
            "status": "success",
            "message": f"Updated model for slice {slice_id}",
            "global_model": [param.tolist() for param in aggregator.get_global_model()]
        })
    return jsonify({"status": "error", "message": "Missing required data"}), 400

@app.route('/get_model', methods=['GET'])
def get_model():
    return jsonify({
        "status": "success",
        "global_model": [param.tolist() for param in aggregator.get_global_model()]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 