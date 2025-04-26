FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set non-interactive installation and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy environment.yml and other necessary files
COPY environment.yml .
COPY . .

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Create activation script
RUN echo "#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate fcdl\nexec \"\$@\"" > /entry.sh \
    && chmod +x /entry.sh

# Set up entry point
ENTRYPOINT ["/entry.sh"]
CMD ["bash"] 