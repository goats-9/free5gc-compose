import subprocess
from ftplib import FTP
import random
import time

from constants import *

# Functions to carry out benign traffic

def do_iperf_tcp(src_ip: str, dst_ip: str | None = TARGET_ADDR):
    """
    Run iperf command on the given IP address with specified flag and time
    limit.
    
    Parameters:
        src_ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Generate random timeout
    tm = random.randint(1, IPERF_TIME)
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"iperf -c {dst_ip} -t {tm}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

def do_iperf_udp(src_ip: str, dst_ip: str | None = TARGET_ADDR):
    """
    Run iperf command with UDP packets on the given IP address with specified 
    flag and time limit.
    
    Parameters:
        src_ip (str): IP address to run the command from.
    
    Returns
        None
    """
    # Generate random timeout
    tm = random.randint(1, IPERF_TIME)
    # Run command
    subprocess.run(
        [f"./nr-binder", src_ip, f"iperf -c {dst_ip} -u -t {tm}"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

def _ftp_discard(_):
    pass

def do_ftp(src_ip: str, dst_ip: str = TARGET_ADDR):
    """
    Run FTP command on the given IP address.
    
    Parameters:
        src_ip (str): IP address to run the command on.
    
    Returns
        None
    """
    # Create FTP connection
    with FTP() as ftp:
        # Connect to host
        ftp.connect(host=dst_ip, port=FTP_PORT, source_address=(src_ip, 0))
        # Login to FTP server
        ftp.login(user=FTP_USER, passwd=FTP_PASS)
        # Get list of files in the directory
        files = ftp.nlst()
        # Choose a random file
        file = random.choice(files)
        # Download the file
        ftp.retrbinary(f"RETR {file}", _ftp_discard)

def do_get(src_ip: str, dst_ip: str | None = None):
    """
    Run HTTP GET on a random site from the given IP address.
    
    Parameters:
        src_ip (str): IP address to run GET from.
    
    Returns
        None
    """
    # Get random site
    if dst_ip is None:
        dst_ip = random.choice(SITES)
    
    # Run command to GET site with requests
    subprocess.run(
        [f"./nr-binder", src_ip, f"wget -p {dst_ip} -O /dev/null"],
        capture_output=True,
        cwd=BINDER_DIR,
    )

BENIGN_LIST=(
    do_iperf_tcp,
    do_iperf_udp,
    do_ftp,
    do_get,
    # Add more benign functions here
)

def benign_ue(ip: str) -> None:
    """
    Generate benign UE traffic.
    
    Parameters:
        ip (str): IP address to generate traffic for.
    
    Returns:
        None
    """
    try:
        while True:
            # Randomly select a benign function
            func = random.choice(BENIGN_LIST)
            # Call the function with the IP address
            func(ip)
            # Sleep for a random interval
            time.sleep(random.randint(1, SLEEP_TIME))
    except Exception as e:
        print(f"Error in benign_ue, IP address {ip}: {e}")