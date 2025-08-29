# Base: RunPod's PyTorch 2.1 image already has CUDA runtime + PyTorch.
# Swap to the exact tag you use on RunPod, e.g. runpod/pytorch:2.1.0-cuda12.1-runtime
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config libssl-dev curl git python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y && \
    /root/.cargo/bin/rustup default stable
ENV PATH="/root/.cargo/bin:${PATH}"

# (Optional) set CUDA-related envs commonly expected by build scripts
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Build moshi-server (CUDA feature)
RUN cargo install --features cuda moshi-server

# Python deps you used
RUN pip install --no-cache-dir huggingface_hub safetensors moshi pydantic

# Bring in the config from the repo (only the config you need to run)
# If you need the full repo (e.g., assets), clone and COPY selectively instead.
WORKDIR /app
RUN git clone https://github.com/kyutai-labs/delayed-streams-modeling /tmp/dsm && \
    mkdir -p /app/configs && \
    cp /tmp/dsm/configs/config-tts.toml /app/configs/ && \
    rm -rf /tmp/dsm

# Expose the Moshi port
EXPOSE 8000

# Add a lightweight start script (see below)
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Default command: launch moshi-server worker
CMD ["/usr/local/bin/start.sh"]

