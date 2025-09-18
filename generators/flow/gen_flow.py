from nfstream import NFStreamer, NFPlugin
from scapy.layers.inet import IP, UDP
from scapy.contrib.pfcp import PFCP, PFCPmessageType
import sys
import argparse
import requests
import json

IFACE='eth0'
ROWS=50000

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


parser = argparse.ArgumentParser(description="Generate PFCP flow data")
parser.add_argument(
    "-r", "--rows", type=int, default=ROWS, help="Number of rows to generate"
)
parser.add_argument(
    "-i", "--iface", type=str, default=IFACE, help="Network interface to capture traffic"
)
parser.add_argument(
    "-m", "--mode", type=str, default="local", choices=["local", "remote"], help="Mode of operation"
)
parser.add_argument(
    "-f", "--file", type=str, help="Output file path (for local mode)"
)
parser.add_argument(
    "-u", "--url", type=str, help="URL to send flows to (for remote mode)"
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "remote" and not args.url:
        parser.error("URL is required in remote mode")
    if args.mode == "local":
        streamer = NFStreamer(
            source=IFACE,
            active_timeout=10,
            idle_timeout=1,
            max_nflows=ROWS,
            udps=PFCPFlowGenerator(),
            statistical_analysis=True
        ).to_csv(
            path=args.file
        )
    else:
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
            requests.post(args.url, data=json.dumps(flow_dict), headers={
                'Content-Type': 'application/json'
            })