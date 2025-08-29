# CUDA 11.8 devel image from RunPod (has headers/toolchain needed to compile with --features cuda)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config libssl-dev curl git python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y && /root/.cargo/bin/rustup default stable
ENV PATH="/root/.cargo/bin:${PATH}"

# CUDA env (11.8)
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Build moshi-server with CUDA
RUN cargo install --features cuda moshi-server

# Python deps
RUN pip install --no-cache-dir huggingface_hub safetensors moshi pydantic

# Bring in config
WORKDIR /app
RUN git clone https://github.com/kyutai-labs/delayed-streams-modeling /tmp/dsm && \
    mkdir -p /app/configs && \
    cp /tmp/dsm/configs/config-tts.toml /app/configs/ && \
    rm -rf /tmp/dsm

# Port & startup
EXPOSE 8000
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh
CMD ["/usr/local/bin/start.sh"]
