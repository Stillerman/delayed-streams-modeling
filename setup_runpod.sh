#!/bin/bash

  

# Moshi Server Installation Script

# This script installs Moshi server with CUDA support

  

set -e  # Exit on any error

  

echo "Starting Moshi Server installation..."

  

# Update system packages

echo "Updating system packages..."

apt update && apt upgrade -y

  

# Install system dependencies

echo "Installing system dependencies..."

apt install -y cmake pkg-config libssl-dev

  

# Install Rust if not already installed

echo "Installing Rust..."

if ! command -v rustc &> /dev/null; then

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    source "$HOME/.cargo/env"

else

    echo "Rust is already installed"

fi

  

# Ensure cargo is in PATH

export PATH="$HOME/.cargo/bin:$PATH"

source "$HOME/.cargo/env"

  

# Clone the repository

echo "Cloning delayed-streams-modeling repository..."

if [ ! -d "delayed-streams-modeling" ]; then

    git clone https://github.com/kyutai-labs/delayed-streams-modeling/

else

    echo "Repository already exists, skipping clone"

fi

  

cd delayed-streams-modeling/

  

# Install moshi-server with CUDA support

echo "Installing moshi-server with CUDA support..."

cargo install --features cuda moshi-server

  

# Install Python dependencies

echo "Installing Python dependencies..."

pip install huggingface_hub safetensors moshi pydantic

  

echo "Installation complete!"

echo ""

echo "To run the server:"

echo "cd delayed-streams-modeling/"

echo "moshi-server worker --config configs/config-tts.toml --port 8000"

echo ""

echo "Make sure you have CUDA properly installed and configured for GPU acceleration."
