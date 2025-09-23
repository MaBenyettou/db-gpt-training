FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git python3 python3-pip build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone llama.cpp for GGUF conversion
RUN git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp && \
    cd /workspace/llama.cpp && make quantize

COPY scripts/ scripts/
COPY configs/ configs/
COPY data/ data/

CMD ["bash"]
