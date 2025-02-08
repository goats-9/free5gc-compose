from nfstream import NFPlugin, NFStreamer
from scapy.layers.inet import IP, TCP, UDP
from scapy.contrib.pfcp import PFCP, PFCPmessageType
import pandas as pd
import sys

IFACE='eth0'

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

ROWS=50000

if __name__ == "__main__":
    streamer = NFStreamer(
        source=IFACE,
        active_timeout=10,
        idle_timeout=1,
        max_nflows=ROWS,
        udps=PFCPFlowGenerator(),
        statistical_analysis=True
    ).to_csv(
        path=sys.argv[1],
    )