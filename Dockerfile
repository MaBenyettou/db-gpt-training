FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git python3 python3-pip build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /workspace

# Set Hugging Face cache directory
ENV HF_HOME=/workspace/.cache/huggingface

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install mistral_common

# Clone llama.cpp for GGUF conversion
RUN git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp && \
    cd /workspace/llama.cpp && mkdir build && cd build && \
    cmake .. -DLLAMA_CURL=OFF && make -j$(nproc)

# Pre-download model (optional – avoids redownloading each run)
# Replace with the model you want
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-Coder-3B-Instruct', local_dir='/workspace/models/Qwen2.5')"

# Copy your training scripts/configs/data
COPY scripts/ scripts/
COPY configs/ configs/
COPY data/ data/

CMD ["bash"]
