# ---------- Builder (smaller) ----------
    FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

    ARG CUDA_COMPUTE_CAP=89
    ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
    ENV DEBIAN_FRONTEND=noninteractive
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
          build-essential cmake pkg-config libssl-dev curl git ca-certificates \
          python3-minimal && \
        rm -rf /var/lib/apt/lists/*
    
    # Rust
    RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y && /root/.cargo/bin/rustup default stable
    ENV PATH="/root/.cargo/bin:${PATH}"
    ENV CUDA_HOME=/usr/local/cuda
    ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
    
    # Build moshi-server with CUDA
    RUN cargo install --features cuda moshi-server
    
    # Grab just the config(s) you need
    WORKDIR /build
    RUN git clone --depth=1 https://github.com/kyutai-labs/delayed-streams-modeling dsm && \
        mkdir -p /build/configs && \
        cp dsm/configs/config-tts.toml /build/configs/
    
    # ---------- Runtime (RunPod) ----------
    FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04
    
    ENV DEBIAN_FRONTEND=noninteractive
    # python3-pip is present on most tags, but install if missing:
    RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && \
        rm -rf /var/lib/apt/lists/*
    
    # Keep pip lean
    ENV PIP_NO_CACHE_DIR=1
    RUN pip install --no-cache-dir huggingface_hub safetensors moshi pydantic && \
        python3 -m pip cache purge || true
    
    # Copy only the compiled binary + configs
    COPY --from=builder /root/.cargo/bin/moshi-server /usr/local/bin/moshi-server
    COPY --from=builder /build/configs /app/configs
    
    ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
    WORKDIR /app
    EXPOSE 8000
    
    # Start script
    COPY start.sh /usr/local/bin/start.sh
    RUN chmod +x /usr/local/bin/start.sh
    CMD ["/usr/local/bin/start.sh"]
    