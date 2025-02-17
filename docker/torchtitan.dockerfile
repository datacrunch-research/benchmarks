# Use NVIDIA's official CUDA 12.4 base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ARG HF_TOKEN    
ENV HF_TOKEN=${HF_TOKEN}

# makes sure the shell used for subsequent RUN commands is exactly Bash, as located in /bin.
SHELL ["/bin/bash", "-c"]

# Install dependencies
RUN apt-get update && apt-get install -y \
    fzf \
    ripgrep \
    nvtop \
    sudo \
    kmod \
    wget \
    vim \
    git \
    curl \
    bzip2 \
    ca-certificates 
    # Cleanup command to remove the apt cache and reduce the image size: # IMPORTANT: Enforces using sudo apt update when entering the container
    #&& rm -rf /var/lib/apt/lists/*

# Cloning the repo
RUN git clone https://github.com/pytorch/torchtitan
RUN pip install -r torchtitan/requirements.txt
# CARE: using cu126 instead 124
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# Change to the repo directory using WORKDIR
WORKDIR /workspace/torchtitan

# Download the tokenizer
RUN python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"


# docker build . --build-arg HF_TOKEN="$HF_TOKEN" -t torchtitan_cuda126_torch26
# docker run --gpus all --shm-size 32g --network=host -v /mnt/local_nvme/rodri:/root/.cache/huggingface --name torchtitan_workload -it --rm --ipc=host torchtitan_cuda126_torch26 bash -c 'CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh'
# docker run --gpus all --shm-size 32g --network=host -v /mnt/local_nvme/rodri:/root/.cache/huggingface --name torchtitan_workload -it --rm --ipc=host torchtitan_cuda126_torch26 bash