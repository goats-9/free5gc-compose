# Imports
from pathlib import Path
import argparse
import ruamel.yaml
from ruamel.yaml import YAML
import ruamel.yaml.scalarint

# Constants
ROOT_PATH = str(Path(__file__).parents[2].absolute())

IP_OFFSET = 60
IMSI_OFFSET = 208931000000001

CORE_TEMPLATE="""services:
  free5gc-upf:
    container_name: upf
    # image: free5gc/upf:v4.0.0
    build: nf_upf
    command: bash -c "./upf-iptables.sh && ./upf -c ./config/upfcfg.yaml"
    volumes:
      - ./config/upfcfg.yaml:/free5gc/config/upfcfg.yaml
      - ./config/upf-iptables.sh:/free5gc/upf-iptables.sh
      - ./generators/flow:/free5gc/flow
    cap_add:
      - NET_ADMIN
    networks:
      privnet:
        aliases:
          - upf.free5gc.org

  db:
    container_name: mongodb
    image: mongo:3.6.8
    command: mongod --port 27017
    expose:
      - "27017"
    volumes:
      - dbdata:/data/db
    networks:
      privnet:
        aliases:
          - db

  free5gc-nrf:
    container_name: nrf
    image: free5gc/nrf:v4.0.0
    command: ./nrf -c ./config/nrfcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/nrfcfg.yaml:/free5gc/config/nrfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      DB_URI: mongodb://db/free5gc
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - nrf.free5gc.org
    depends_on:
      - db

  free5gc-amf:
    container_name: amf
    image: free5gc/amf:v4.0.0
    command: ./amf -c ./config/amfcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/amfcfg-slice.yaml:/free5gc/config/amfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        ipv4_address: 10.100.200.16
        aliases:
          - amf.free5gc.org
    depends_on:
      - free5gc-nrf

  free5gc-ausf:
    container_name: ausf
    image: free5gc/ausf:v4.0.0
    command: ./ausf -c ./config/ausfcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/ausfcfg.yaml:/free5gc/config/ausfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - ausf.free5gc.org
    depends_on:
      - free5gc-nrf

  free5gc-nssf:
    container_name: nssf
    image: free5gc/nssf:v4.0.0
    command: ./nssf -c ./config/nssfcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/nssfcfg.yaml:/free5gc/config/nssfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - nssf.free5gc.org
    depends_on:
      - free5gc-nrf

  free5gc-pcf:
    container_name: pcf
    image: free5gc/pcf:v4.0.0
    command: ./pcf -c ./config/pcfcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/pcfcfg.yaml:/free5gc/config/pcfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - pcf.free5gc.org
    depends_on:
      - free5gc-nrf

  free5gc-smf:
    container_name: smf
    image: free5gc/smf:v4.0.0
    command: ./smf -c ./config/smfcfg.yaml -u ./config/uerouting.yaml
    expose:
      - "8000"
    volumes:
      - ./config/smfcfg.yaml:/free5gc/config/smfcfg.yaml
      - ./config/uerouting.yaml:/free5gc/config/uerouting.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - smf.free5gc.org
    depends_on:
      - free5gc-nrf
      - free5gc-upf

  free5gc-udm:
    container_name: udm
    image: free5gc/udm:v4.0.0
    command: ./udm -c ./config/udmcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/udmcfg.yaml:/free5gc/config/udmcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - udm.free5gc.org
    depends_on:
      - db
      - free5gc-nrf

  free5gc-udr:
    container_name: udr
    image: free5gc/udr:v4.0.0
    command: ./udr -c ./config/udrcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/udrcfg.yaml:/free5gc/config/udrcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      DB_URI: mongodb://db/free5gc
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - udr.free5gc.org
    depends_on:
      - db
      - free5gc-nrf

  free5gc-chf:
    container_name: chf
    image: free5gc/chf:v4.0.0
    command: ./chf -c ./config/chfcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/chfcfg.yaml:/free5gc/config/chfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      DB_URI: mongodb://db/free5gc
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - chf.free5gc.org
    depends_on:
      - db
      - free5gc-nrf
      - free5gc-webui

  free5gc-n3iwf:
    container_name: n3iwf
    image: free5gc/n3iwf:v4.0.0
    command: ./n3iwf -c ./config/n3iwfcfg.yaml
    volumes:
      - ./config/n3iwfcfg.yaml:/free5gc/config/n3iwfcfg.yaml
      - ./config/n3iwf-ipsec.sh:/free5gc/n3iwf-ipsec.sh
    environment:
      GIN_MODE: release
    cap_add:
      - NET_ADMIN
    networks:
      privnet:
        ipv4_address: 10.100.200.15
        aliases:
          - n3iwf.free5gc.org
    depends_on:
      - free5gc-amf
      - free5gc-smf
      - free5gc-upf

  free5gc-tngf:
    container_name: tngf
    image: free5gc/tngf:latest
    command: ./tngf -c ./config/tngfcfg.yaml
    volumes:
      - ./config/tngfcfg.yaml:/free5gc/config/tngfcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    cap_add:
      - NET_ADMIN
    network_mode: host
    depends_on:
      - free5gc-amf
      - free5gc-smf
      - free5gc-upf

  free5gc-nef:
    container_name: nef
    image: free5gc/nef:latest
    command: ./nef -c ./config/nefcfg.yaml
    expose:
      - "8000"
    volumes:
      - ./config/nefcfg.yaml:/free5gc/config/nefcfg.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - nef.free5gc.org
    depends_on:
      - db
      - free5gc-nrf

  free5gc-webui:
    container_name: webui
    image: free5gc/webui:v4.0.0
    command: ./webui -c ./config/webuicfg.yaml
    expose:
      - "2121"
    volumes:
      - ./config/webuicfg.yaml:/free5gc/config/webuicfg.yaml
    environment:
      - GIN_MODE=release
    networks:
      privnet:
        aliases:
          - webui
    ports:
      - "5000:5000"
      - "2122:2122"
      - "2121:2121"
    depends_on:
      - db
      - free5gc-nrf

  ueransim:
    container_name: ueransim
    build:
      context: ueransim
      dockerfile: Dockerfile
    # image: free5gc/ueransim:latest
    command: ./nr-gnb -c ./config/gnbcfg.yaml
    # UE configurations to be added here
    volumes:
      - ./config/gnbcfg-slice.yaml:/ueransim/config/gnbcfg.yaml
      - ./generators/traffic:/ueransim/traffic
    cap_add:
      - NET_ADMIN
    devices:
      - "/dev/net/tun"
    networks:
      privnet:
        aliases:
          - gnb.free5gc.org
    depends_on:
      - free5gc-amf
      - free5gc-upf

  n3iwue:
    container_name: n3iwue
    image: free5gc/n3iwue:latest
    command: bash -c "ip route del default && ip route add default via 10.100.200.1 dev eth0 metric 203 && sleep infinity"
    volumes:
      - ./config/n3uecfg.yaml:/n3iwue/config/n3ue.yaml
    cap_add:
      - NET_ADMIN
    devices:
      - "/dev/net/tun"
    networks:
      privnet:
        ipv4_address: 10.100.200.203
        aliases:
          - n3ue.free5gc.org
    depends_on:
      - free5gc-n3iwf

  target:
    build: target
    container_name: target
    ports:
      - "5001:5001"
      - "8080:8000"
      - "8222:22"
      - "8121:2121"
    volumes:
      - ./target/ftp:/var/ftp
    networks:
      privnet:
        aliases:
          - target.free5gc.org

networks:
  privnet:
    ipam:
      driver: default
      config:
        - subnet: 10.100.200.0/24
    driver_opts:
      com.docker.network.bridge.name: br-free5gc

volumes:
  dbdata:
"""

SLICE_TEMPLATE="""services:
  free5gc-upf-{i}:
    container_name: upf-{i}
    # image: free5gc/upf:v4.0.0
    build: nf_upf
    command: bash -c "./upf-iptables.sh && ./upf -c ./config/upfcfg.yaml"
    volumes:
      - ./config/upfcfg-{i}.yaml:/free5gc/config/upfcfg.yaml
      - ./config/upf-iptables.sh:/free5gc/upf-iptables.sh
      - ./generators/flow:/free5gc/flow
    cap_add:
      - NET_ADMIN
    networks:
      privnet:
        aliases:
          - upf-{i}.free5gc.org

  free5gc-smf-{i}:
    container_name: smf-{i}
    image: free5gc/smf:v4.0.0
    command: ./smf -c ./config/smfcfg.yaml -u ./config/uerouting.yaml
    expose:
      - "8000"
    volumes:
      - ./config/smfcfg-{i}.yaml:/free5gc/config/smfcfg.yaml
      - ./config/uerouting-{i}.yaml:/free5gc/config/uerouting.yaml
      - ./cert:/free5gc/cert
    environment:
      GIN_MODE: release
    networks:
      privnet:
        aliases:
          - smf-{i}.free5gc.org
    depends_on:
      - free5gc-nrf
      - free5gc-upf-{i}
"""

AMFCFG_TEMPLATE = """
info:
  version: 1.0.9
  description: AMF initial local configuration

configuration:
  amfName: AMF # the name of this AMF
  ngapIpList:  # the IP list of N2 interfaces on this AMF
    - amf.free5gc.org
  ngapPort: 38412 # the SCTP port listened by NGAP
  sbi: # Service-based interface information
    scheme: http # the protocol for sbi (http or https)
    registerIPv4: amf.free5gc.org # IP used to register to NRF
    bindingIPv4: amf.free5gc.org  # IP used to bind the service
    port: 8000 # port used to bind the service
    tls: # the local path of TLS key
      pem: cert/amf.pem # AMF TLS Certificate
      key: cert/amf.key # AMF TLS Private key
  serviceNameList: # the SBI services provided by this AMF, refer to TS 29.518
    - namf-comm # Namf_Communication service
    - namf-evts # Namf_EventExposure service
    - namf-mt   # Namf_MT service
    - namf-loc  # Namf_Location service
    - namf-oam  # OAM service
  servedGuamiList: # Guami (Globally Unique AMF ID) list supported by this AMF
    # <GUAMI> = <MCC><MNC><AMF ID>
    - plmnId: # Public Land Mobile Network ID, <PLMN ID> = <MCC><MNC>
        mcc: 208 # Mobile Country Code (3 digits string, digit: 0~9)
        mnc: 93 # Mobile Network Code (2 or 3 digits string, digit: 0~9)
      amfId: cafe00 # AMF identifier (3 bytes hex string, range: 000000~FFFFFF)
  supportTaiList:  # the TAI (Tracking Area Identifier) list supported by this AMF
    - plmnId: # Public Land Mobile Network ID, <PLMN ID> = <MCC><MNC>
        mcc: 208 # Mobile Country Code (3 digits string, digit: 0~9)
        mnc: 93 # Mobile Network Code (2 or 3 digits string, digit: 0~9)
      tac: 000001 # Tracking Area Code (3 bytes hex string, range: 000000~FFFFFF)
  plmnSupportList: # the PLMNs (Public land mobile network) list supported by this AMF
    - plmnId: # Public Land Mobile Network ID, <PLMN ID> = <MCC><MNC>
        mcc: 208 # Mobile Country Code (3 digits string, digit: 0~9)
        mnc: 93 # Mobile Network Code (2 or 3 digits string, digit: 0~9)
      snssaiList: # the S-NSSAI (Single Network Slice Selection Assistance Information) list supported by this AMF
  supportDnnList:  # the DNN (Data Network Name) list supported by this AMF
    - internet
  nrfUri: http://nrf.free5gc.org:8000 # a valid URI of NRF
  nrfCertPem: cert/nrf.pem
  security:  # NAS security parameters
    integrityOrder: # the priority of integrity algorithms
      - NIA2
      # - NIA0
    cipheringOrder: # the priority of ciphering algorithms
      - NEA0
      # - NEA2
  networkName:  # the name of this core network
    full: free5GC
    short: free
  ngapIE: # Optional NGAP IEs
    mobilityRestrictionList: # Mobility Restriction List IE, refer to TS 38.413
      enable: true # append this IE in related message or not
    maskedIMEISV: # Masked IMEISV IE, refer to TS 38.413
      enable: true # append this IE in related message or not
    redirectionVoiceFallback: # Redirection Voice Fallback IE, refer to TS 38.413
      enable: false # append this IE in related message or not
  nasIE: # Optional NAS IEs
    networkFeatureSupport5GS: # 5gs Network Feature Support IE, refer to TS 24.501
      enable: true # append this IE in Registration accept or not
      length: 1 # IE content length (uinteger, range: 1~3)
      imsVoPS: 0 # IMS voice over PS session indicator (uinteger, range: 0~1)
      emc: 0 # Emergency service support indicator for 3GPP access (uinteger, range: 0~3)
      emf: 0 # Emergency service fallback indicator for 3GPP access (uinteger, range: 0~3)
      iwkN26: 0 # Interworking without N26 interface indicator (uinteger, range: 0~1)
      mpsi: 0 # MPS indicator (uinteger, range: 0~1)
      emcN3: 0 # Emergency service support indicator for Non-3GPP access (uinteger, range: 0~1)
      mcsi: 0 # MCS indicator (uinteger, range: 0~1)
  t3502Value: 720  # timer value (seconds) at UE side
  t3512Value: 3600 # timer value (seconds) at UE side
  non3gppDeregTimerValue: 3240 # timer value (seconds) at UE side
  # retransmission timer for paging message
  t3513:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  # retransmission timer for NAS Deregistration Request message
  t3522:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  # retransmission timer for NAS Registration Accept message
  t3550:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  # retransmission timer for NAS Configuration Update Command message
  t3555:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  # retransmission timer for NAS Authentication Request/Security Mode Command message
  t3560:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  # retransmission timer for NAS Notification message
  t3565:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  # retransmission timer for NAS Identity Request message
  t3570:
    enable: true     # true or false
    expireTime: 6s   # default is 6 seconds
    maxRetryTimes: 4 # the max number of retransmission
  locality: area1 # Name of the location where a set of AMF, SMF, PCF and UPFs are located
  sctp: # set the sctp server setting <optinal>, once this field is set, please also add maxInputStream, maxOsStream, maxAttempts, maxInitTimeOut
    numOstreams: 3 # the maximum out streams of each sctp connection
    maxInstreams: 5 # the maximum in streams of each sctp connection
    maxAttempts: 2 # the maximum attempts of each sctp connection
    maxInitTimeout: 2 # the maximum init timeout of each sctp connection
  defaultUECtxReq: false # the default value of UE Context Request to decide when triggering Initial Context Setup procedure

logger: # log output setting
  enable: true # true or false
  level: info # how detailed to output, value: trace, debug, info, warn, error, fatal, panic
  reportCaller: false # enable the caller report or not, value: true or false
"""

GNBCFG_TEMPLATE = """mcc: "208" # Mobile Country Code value
mnc: "93" # Mobile Network Code value (2 or 3 digits)

nci: "0x000000010" # NR Cell Identity (36-bit)
idLength: 32 # NR gNB ID length in bits [22...32]
tac: 1 # Tracking Area Code

linkIp: 127.0.0.1 # gNB's local IP address for Radio Link Simulation (Usually same with local IP)
ngapIp: gnb.free5gc.org # gNB's local IP address for N2 Interface (Usually same with local IP)
gtpIp: gnb.free5gc.org # gNB's local IP address for N3 Interface (Usually same with local IP)

# List of AMF address information
amfConfigs:
  - address: amf.free5gc.org
    port: 38412

# List of supported S-NSSAIs by this gNB
slices:

# Indicates whether or not SCTP stream number errors should be ignored.
ignoreStreamIds: true
"""

SMFCFG_TEMPLATE = """info:
  version: 1.0.7
  description: SMF initial local configuration

configuration:
  smfName: SMF # the name of this SMF
  sbi: # Service-based interface information
    scheme: http # the protocol for sbi (http or https)
    registerIPv4: smf-{i}.free5gc.org # IP used to register to NRF
    bindingIPv4: smf-{i}.free5gc.org # IP used to bind the service
    port: 8000 # Port used to bind the service
    tls: # the local path of TLS key
      key: cert/smf.key # SMF TLS Certificate
      pem: cert/smf.pem # SMF TLS Private key
  serviceNameList: # the SBI services provided by this SMF, refer to TS 29.502
    - nsmf-pdusession # Nsmf_PDUSession service
    - nsmf-event-exposure # Nsmf_EventExposure service
    - nsmf-oam # OAM service
  snssaiInfos: # the S-NSSAI (Single Network Slice Selection Assistance Information) list supported by this AMF
    - sNssai: # S-NSSAI (Single Network Slice Selection Assistance Information)
        sst: 1 # Slice/Service Type (uinteger, range: 0~255)
        sd: {i:06} # Slice Differentiator (3 bytes hex string, range: 000000~FFFFFF)
      dnnInfos: # DNN information list
        - dnn: internet # Data Network Name
          dns: # the IP address of DNS
            ipv4: 8.8.8.8
            ipv6: 2001:4860:4860::8888
  plmnList: # the list of PLMN IDs that this SMF belongs to (optional, remove this key when unnecessary)
    - mcc: 208 # Mobile Country Code (3 digits string, digit: 0~9)
      mnc: 93 # Mobile Network Code (2 or 3 digits string, digit: 0~9)
  locality: area1 # Name of the location where a set of AMF, SMF, PCF and UPFs are located
  pfcp: # the IP address of N4 interface on this SMF (PFCP)
    # addr config is deprecated in smf config v1.0.3, please use the following config
    nodeID: smf-{i}.free5gc.org # the Node ID of this SMF
    listenAddr: smf-{i}.free5gc.org # the IP/FQDN of N4 interface on this SMF (PFCP)
    externalAddr: smf-{i}.free5gc.org # the IP/FQDN of N4 interface on this SMF (PFCP)
  userplaneInformation: # list of userplane information
    upNodes: # information of userplane node (AN or UPF)
      gNB: # the name of the node
        type: AN # the type of the node (AN or UPF)
      UPF-{i}: # the name of the node
        type: UPF # the type of the node (AN or UPF)
        nodeID: upf-{i}.free5gc.org # the Node ID of this UPF
        addr: upf-{i}.free5gc.org # the IP/FQDN of N4 interface on this UPF (PFCP)
        sNssaiUpfInfos: # S-NSSAI information list for this UPF
          - sNssai: # S-NSSAI (Single Network Slice Selection Assistance Information)
              sst: 1 # Slice/Service Type (uinteger, range: 0~255)
              sd: {i:06} # Slice Differentiator (3 bytes hex string, range: 000000~FFFFFF)
            dnnUpfInfoList: # DNN information list for this S-NSSAI
              - dnn: internet
                pools:
                  - cidr: 10.{j}.0.0/16
                staticPools:
                  - cidr: 10.{j}.100.0/24
        interfaces: # Interface list for this UPF
          - interfaceType: N3 # the type of the interface (N3 or N9)
            endpoints: # the IP address of this N3/N9 interface on this UPF
              - upf-{i}.free5gc.org
            networkInstances: # Data Network Name (DNN)
              - internet
    links: # the topology graph of userplane, A and B represent the two nodes of each link
      - A: gNB
        B: UPF-{i}
  # retransmission timer for pdu session modification command
  t3591:
    enable: true # true or false
    expireTime: 16s # default is 6 seconds
    maxRetryTimes: 3 # the max number of retransmission
  # retransmission timer for pdu session release command
  t3592:
    enable: true # true or false
    expireTime: 16s # default is 6 seconds
    maxRetryTimes: 3 # the max number of retransmission
  nrfUri: http://nrf.free5gc.org:8000 # a valid URI of NRF
  nrfCertPem: cert/nrf.pem # NRF Certificate
  urrPeriod: 10 # default usage report period in seconds
  urrThreshold: 1000 # default usage report threshold in bytes
  requestedUnit: 1000
logger: # log output setting
  enable: true # true or false
  level: info # how detailed to output, value: trace, debug, info, warn, error, fatal, panic
  reportCaller: false # enable the caller report or not, value: true or false
"""

UPFCFG_TEMPLATE = """version: 1.0.3
description: UPF initial local configuration

# The listen IP and nodeID of the N4 interface on this UPF (Can't set to 0.0.0.0)
pfcp:
  addr: upf-{i}.free5gc.org # IP addr for listening
  nodeID: upf-{i}.free5gc.org # External IP or FQDN can be reached
  retransTimeout: 1s # retransmission timeout
  maxRetrans: 3 # the max number of retransmission

gtpu:
  forwarder: gtp5g
  # The IP list of the N3/N9 interfaces on this UPF
  # If there are multiple connection, set addr to 0.0.0.0 or list all the addresses
  ifList:
    - addr: upf-{i}.free5gc.org
      type: N3

# The DNN list supported by UPF
dnnList:
  - dnn: internet # Data Network Name
    cidr: 10.{j}.0.0/16 # Classless Inter-Domain Routing for assigned IPv4 pool of UE
    sNssaiUpfInfos:
      - sNssai:
          sst: 1
          sd: {i:06}  # Slice 1   

logger: # log output setting
  enable: true # true or false
  level: debug # how detailed to output, value: trace, debug, info, warn, error, fatal, panic
  reportCaller: false # enable the caller report or not, value: true or false
"""

UEROUTING_TEMPLATE = """info:
  version: 1.0.7
  description: Routing information for UE

ueRoutingInfo: # the list of UE routing information
  UE-{i}: # Group Name
    members:
    - imsi-{j} # Subscription Permanent Identifier of the UE
    topology: # Network topology for this group (Uplink: A->B, Downlink: B->A)
    # default path derived from this topology
    # node name should be consistent with smfcfg.yaml
      - A: gNB
        B: UPF-{i}
    specificPath:
      - dest: 8.8.8.8/32 # the destination IP address on Data Network (DN)
        # the order of UPF nodes in this path. We use the UPF's name to represent each UPF node.
        # The UPF's name should be consistent with smfcfg.yaml
        path: [UPF-{i}]
"""

UECFG_TEMPLATE = """# IMSI number of the UE. IMSI = [MCC|MNC|MSISDN] (In total 15 digits)
supi: "imsi-{j}"
# Mobile Country Code value of HPLMN
mcc: "208"
# Mobile Network Code value of HPLMN (2 or 3 digits)
mnc: "93"

# Permanent subscription key
key: "8baf473f2f8fd09487cccbd7097c6862"
# Operator code (OP or OPC) of the UE
op: "8e27b6af0e692e750f32667a3b14605d"
# This value specifies the OP type and it can be either 'OP' or 'OPC'
opType: "OP"
# Authentication Management Field (AMF) value
amf: "8000"
# IMEI number of the device. It is used if no SUPI is provided
imei: "356938035643803"
# IMEISV number of the device. It is used if no SUPI and IMEI is provided
imeiSv: "4370816125816151"

# List of gNB IP addresses for Radio Link Simulation
gnbSearchList:
  - 127.0.0.1
  - gnb.free5gc.org

# UAC Access Identities Configuration
uacAic:
  mps: false
  mcs: false

# UAC Access Control Class
uacAcc:
  normalClass: 0
  class11: false
  class12: false
  class13: false
  class14: false
  class15: false

# Initial PDU sessions to be established
sessions:
  - type: "IPv4"
    apn: "internet"
    slice:
      sst: 0x01
      sd: 0x{i:06x}
  
# Configured NSSAI for this UE by HPLMN
configured-nssai:
  - sst: 0x01
    sd: 0x{i:06x}

# Default Configured NSSAI for this UE
default-nssai:
  - sst: 1
    sd: 1

# Supported integrity algorithms by this UE
integrity:
  IA1: true
  IA2: true
  IA3: true

# Supported encryption algorithms by this UE
ciphering:
  EA1: true
  EA2: true
  EA3: true

# Integrity protection maximum data rate for user plane
integrityMaxRate:
  uplink: "full"
  downlink: "full"
"""

def remove_space_after_colon(s):
    res = []
    for line in s.splitlines(True):
        # Remove space for volumes that contain ue configs
        if 'uecfg' in line:
            res.append(line.replace(': ', ':', 1))
        else:
            res.append(line)
    return ''.join(res)

def gen_core(n: int) -> None:
    """
    Generate docker-compose file for the core.
    Parameters:
        n (int): Number of slices.
    Returns:
        None
    """
    with open(f'{ROOT_PATH}/docker-compose-core.yaml', 'w') as f:
        yaml = YAML()
        core_yaml = yaml.load(CORE_TEMPLATE)
        # Append UE configs to ueransim volumes
        uecfg_vols = []
        for i in range(1, n + 1):
            uecfg = {
                f'./config/uecfg-{i}.yaml': f'/ueransim/config/uecfg-{i}.yaml',
            }
            uecfg_vols.append(uecfg)
        # Add volumes
        core_yaml['services']['ueransim']['volumes'].extend(uecfg_vols)
        # Dump the yaml, remove extra whitespace where needed
        yaml.dump(core_yaml, f, transform=remove_space_after_colon)

def gen_compose(i: int) -> None:
    """
    Generate docker-compose file for a slice.
    Parameters:
        i (int): Index of the slice
    Returns:
        None
    """
    with open(f'{ROOT_PATH}/docker-compose-slice-{i}.yaml', 'w') as f:
        # Generate docker-compose file
        f.write(SLICE_TEMPLATE.format(i=i))

def gen_amf(n: int) -> None:
    """
    Generate amf config file
    Parameters:
        n (int): Number of slices
    Returns:
        None
    """
    # Generate gnb config file
    with open(f'{ROOT_PATH}/config/amfcfg-slice.yaml', 'w') as amf_cfg:
        # Load the gnb yaml file
        yaml = YAML()
        amf_yaml = yaml.load(AMFCFG_TEMPLATE)
        # Generate the slices
        slices = []
        for i in range(1, n + 1):
            slice_info = {
                'sst': 1,
                'sd': ruamel.yaml.scalarint.ScalarInt(i, width=6),
            }
            slices.append(slice_info)
        # Add the slices to the gnb yaml file
        amf_yaml['configuration']['plmnSupportList'][0]['snssaiList'] = slices
        # Write the gnb yaml file
        yaml.dump(amf_yaml, amf_cfg)

def gen_gnb(n: int) -> None:
    """
    Generate gnb config file
    Parameters:
        n (int): Number of slices
    Returns:
        None
    """
    # Generate gnb config file
    with open(f'{ROOT_PATH}/config/gnbcfg-slice.yaml', 'w') as gnb_cfg:
        # Load the gnb yaml file
        yaml = YAML()
        gnb_yaml = yaml.load(GNBCFG_TEMPLATE)
        # Generate the slices
        slices = []
        for i in range(1, n + 1):
            slice_info = {
                'sst': ruamel.yaml.scalarint.HexInt(value=1, width=1),
                'sd': ruamel.yaml.scalarint.HexInt(value=i, width=6),
            }
            slices.append(slice_info)
        # Add the slices to the gnb yaml file
        gnb_yaml['slices'] = slices
        # Write the gnb yaml file
        yaml.dump(gnb_yaml, gnb_cfg)

def gen_smf(i: int) -> None:
    """
    Generate SMF config file
    Parameters:
        i (int): Index of the slice
    Returns:
        None
    """
    # Generate smf config file
    with open(f'{ROOT_PATH}/config/smfcfg-{i}.yaml', 'w') as smf_cfg:
        smf_cfg.write(SMFCFG_TEMPLATE.format(i=i, j=IP_OFFSET + i - 1))

def gen_upf(i: int) -> None:
    """
    Generate UPF config file
    Parameters:
        i (int): Index of the slice
    Returns:
        None
    """
    # Generate upf config file
    with open(f'{ROOT_PATH}/config/upfcfg-{i}.yaml', 'w') as upf_cfg:
        upf_cfg.write(UPFCFG_TEMPLATE.format(i=i, j=IP_OFFSET + i - 1))
  
def gen_uerouting(i: int) -> None:
    """
    Generate UEROUTING config file
    Parameters:
        i (int): Index of the slice
    Returns:
        None
    """
    # Generate uerouting config file
    with open(f'{ROOT_PATH}/config/uerouting-{i}.yaml', 'w') as ue_routing:
        ue_routing.write(UEROUTING_TEMPLATE.format(i=i, j=IMSI_OFFSET + (i - 1) * 100))

def gen_uecfg(i: int) -> None:
    """
    Generate UECFG config file
    Parameters:
        i (int): Index of the slice
    Returns:
        None
    """
    # Generate ue config file
    with open(f'{ROOT_PATH}/config/uecfg-{i}.yaml', 'w') as ue_cfg:
        ue_cfg.write(UECFG_TEMPLATE.format(i=i, j=IMSI_OFFSET + (i - 1) * 100))

parser = argparse.ArgumentParser(description='Generate docker-compose files for slices')
parser.add_argument(
    '-n', '--num_slices',
    type=int,
    required=True,
    help='Number of slices to generate'
)
args = parser.parse_args()
if __name__ == '__main__':
    for i in range(1, args.num_slices + 1):
        # Generate files for the slice
        gen_compose(i)
        gen_smf(i)
        gen_upf(i)
        gen_uerouting(i)
        gen_uecfg(i)
    # Generate amf config file
    gen_amf(args.num_slices)
    # Generate gnb config file
    gen_gnb(args.num_slices)
    # Generate the core compose file
    gen_core(args.num_slices)
