FROM ubuntu:latest

# Install dependencies
RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y \
    openssh-server \
    python3 \
    python3-pip \
    python3-venv \
    iperf \
    supervisor

# Enable root login for SSH
RUN mkdir -p /var/run/sshd
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Create python venv and install packages
RUN python3 -m venv /venv
RUN /venv/bin/pip install --upgrade pip
RUN /venv/bin/pip install pyftpdlib

# Create FTP directory
RUN mkdir -p /var/ftp

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start the supervisor daemon
CMD [ "/usr/bin/supervisord" ]
