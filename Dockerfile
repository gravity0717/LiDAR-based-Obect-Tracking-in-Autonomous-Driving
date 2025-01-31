# Base image: Ubuntu 22.04
FROM ubuntu:22.04

# Set timezone to avoid interaction during installation
ENV DEBIAN_FRONTEND=noninteractive


# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    unzip \
    mesa-utils \
    x11-apps \
    libgl1-mesa-dri \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the GitHub repository
RUN git clone https://github.com/gravity0717/LiDAR-based-Obect-Tracking-in-Autonomous-Driving.git

# Change directory to the cloned repository
WORKDIR /workspace/LiDAR-based-Obect-Tracking-in-Autonomous-Driving

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Define KITTI dataset mount point (will be mounted at runtime)
VOLUME [ "/home/poseidon/workspace/dataset/KITTI/raw"]

# Default command: start bash shell
CMD ["bash"]
