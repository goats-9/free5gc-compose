# Imports
from multiprocessing import Pool
import psutil
import argparse
from itertools import repeat
import random
import time

from utils import *
from constants import *

def get_ipv4(addr) -> str:
    """
    Get the IPv4 address from a list of addresses.
    """
    for iface in addr:
        if iface.family == 2:
            return iface.address
    return ""

def dispatch_ue(ip: str, mode: str) -> None:
    try:
        # Dispatch based on mode
        if mode == "benign":
            benign_ue(ip)
        elif mode == "ddos":
            ddos_ue(ip)
        elif mode == "brute_force":
            brute_force_ue(ip)
        elif mode == "pfcp":
            pfcp_ue(ip)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    except Exception as e:
        print(f"Error in dispatch_ue, IP address {ip}: {e}")
        return

parser = argparse.ArgumentParser(description="Run UE process")
parser.add_argument(
    "-m", "--mode", type=str, help="Mode to run", choices=["benign", "ddos", "brute_force", "pfcp"], required=False, default="benign"
)

if __name__ == '__main__':
    args = parser.parse_args()
    # Find interfaces beginning with uesimtun
    ifaces = psutil.net_if_addrs()
    # Get IPv4 addresses for interfaces beginning with uesimtun
    addrs = [get_ipv4(ifaces[i]) for i in ifaces if i.startswith(IFACE_PREFIX)]
    # Create a pool of workers, one for each interface
    with Pool(len(addrs)) as p:
        p.starmap(dispatch_ue, zip(addrs, repeat(args.mode)))