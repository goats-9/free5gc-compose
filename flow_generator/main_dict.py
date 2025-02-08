from nfstream import NFPlugin, NFStreamer
from scapy.layers.inet import IP, UDP
from scapy.contrib.pfcp import PFCP, PFCPmessageType
import requests
import json
import socket

IFACE='enp1s0'
NWDAF_URL=socket.gethostbyaddr('nwdaf-1')
NWDAF_PORT=8080

class PFCPFlowGenerator(NFPlugin):
    def on_init(self, packet, flow):
        # Initialize counters for each flow
        for message_type in PFCPmessageType.values():
            setattr(flow.udps, f"{message_type}_counter", 0)

    def on_update(self, packet, flow):
        # Update counters based on packet direction and PFCP message type
        if packet.protocol != 17 or packet.src_port != 8805 or packet.dst_port != 8805:  # PFCP uses UDP over port 8805
            return

        # Check for PFCP payload
        ip_packet = IP(packet.ip_packet)
        try:
            udp_dgram = ip_packet[UDP]
            payload = udp_dgram[PFCP]
        except:
            return
        if not payload:
            return

        # Extract message type from PFCP header
        try:
            message_type = payload.message_type
            for key, value in PFCPmessageType.items():
                if message_type == key:
                    counter_name = f"{value}_counter"
                    setattr(flow.udps, counter_name, getattr(flow.udps, counter_name) + 1)
                    break
        except IndexError:
            return

ROWS=100000

if __name__ == "__main__":
    streamer = NFStreamer(
        source=IFACE,
        active_timeout=10,
        idle_timeout=1,
        max_nflows=ROWS,
        udps=PFCPFlowGenerator(),
        statistical_analysis=True
    )
    for flow in streamer:
        flow_dict = dict(zip(flow.keys(), flow.values()))
        requests.post(f'http://{NWDAF_URL}:{NWDAF_PORT}', data=json.dumps(flow_dict), headers={
            'Content-Type': 'application/json'
        })