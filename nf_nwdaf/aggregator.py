# aggregator.py
import flwr as fl
from typing import Dict, Any

def aggregator_factory(args: Dict[str, Any]) -> fl.server.aggregator.Aggregator:
    """
    Called by `flower-superlink aggregator.py`.
    Returns an Aggregator instance (FedAvg-like by default).
    """
    # Return standard aggregator (FedAvg-like) for demonstration.
    return fl.server.aggregator.Aggregator()
