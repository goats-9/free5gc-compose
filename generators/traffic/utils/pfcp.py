# Imports
from scapy.contrib.pfcp import PFCP, PFCPSessionEstablishmentRequest, PFCPSessionDeletionRequest
from scapy.contrib.pfcp import PFCPSessionModificationRequest
from scapy.contrib.pfcp import IE_ApplyAction, IE_FAR_Id, IE_UpdateFAR
from scapy.all import send
from scapy.layers.inet import IP, UDP
import random
import subprocess
import time

from constants import *

def session_establishment_request(src_ip: str, dst_ip: str, seid: int):
    pkt = IP(src=src_ip, dst=dst_ip) / \
        UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
        PFCP(
            version=1,
            message_type=50,
            seid=seid
        ) / \
        PFCPSessionEstablishmentRequest()
    send(pkt, count=PFCP_COUNT, inter=PFCP_INTER, verbose=False)

def session_modification_drop_request(src_ip: str, dst_ip: str, seid: int):
    pkt = IP(src=src_ip, dst=dst_ip) / \
        UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
        PFCP(
            version=1,
            message_type=52,
            seid=seid
        ) / \
        PFCPSessionModificationRequest(IE_list=[
            IE_UpdateFAR(IE_list=[
                IE_ApplyAction(DROP=1), 
                IE_FAR_Id(id=1)
            ]),
        ])
    send(pkt, count=PFCP_COUNT, inter=PFCP_INTER, verbose=False)

def session_modification_dupl_request(src_ip: str, dst_ip: str, seid: int):
    pkt = IP(src=src_ip, dst=dst_ip) / \
        UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
        PFCP(
            version=1,
            message_type=52,
            seid=seid
        ) / \
        PFCPSessionModificationRequest(IE_list=[
            IE_UpdateFAR(IE_list=[
                IE_ApplyAction(DUPL=1), 
                IE_FAR_Id(id=1)
            ]),
        ])
    send(pkt, count=PFCP_COUNT, inter=PFCP_INTER, verbose=False)

def session_deletion_request(src_ip: str, dst_ip: str, seid: int):
    pkt = IP(src=src_ip, dst=dst_ip) / \
        UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
        PFCP(
            version=1,
            message_type=54,
            seid=seid
        ) / \
        PFCPSessionDeletionRequest()
    send(pkt, count=PFCP_COUNT, inter=PFCP_INTER, verbose=False)

PFCP_LIST = (
    session_establishment_request,
    session_modification_drop_request,
    session_modification_dupl_request,
    session_deletion_request
)

def pfcp_ue(ip: str) -> None:
    """
    Generate PFCP UE traffic.
    
    Parameters:
        ip (str): IP address to generate traffic for.
    
    Returns:
        None
    """
    try:
        # Get UPF IP address by running traceroute
        res = subprocess.run(["./nr-binder", ip, "traceroute 1.1.1.1"], capture_output=True, text=True, cwd=BINDER_DIR)
        # Get first hop of traceroute as UPF IP
        upf_ip = res.stdout.splitlines()[1].split()[2][1:-1]
        while True:
            # Randomly select a benign function
            func = random.choice(PFCP_LIST)
            # Call the function with the IP address
            func(ip, upf_ip, random.randint(1, 1000))
            # Sleep for a random interval
            time.sleep(random.randint(1, SLEEP_TIME))
    except Exception as e:
        print(f"Error generating PFCP traffic for {ip}: {e}")