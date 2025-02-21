## [tritonbench](https://github.com/pytorch-labs/tritonbench/tree/main)

### local Installation

```bash
git clone https://github.com/pytorch-labs/tritonbench.git
git submodule update --init --recursive
python install.py

python run.py --op gemm
```

For tests and benchmarks, it needs to be installed as a library:

```bash
pip install -e .
```

### **Dockerfile**

Caveats:

- 49.3 GB
- 44 min of build time

```bash
# From root git folder
cd docker
docker build . --file tritonbench-nightly.dockerfile -t tritonbench
```

```bash
docker run --gpus all \
--shm-size 32g \
--network=host \
-v <local_HF_HOME>:<container_HF_HOME> \
--name tritonbench_workload \
-it \
--rm \
--ipc=host \
tritonbench bash
```

No docs → SO docs

### **docs**

Tritonbench is a collection of PyTorch custom operators with example inputs to measure their performance.

- Custom kernels
    - (CUDA, HIP) [kernels](https://github.com/triton-lang/kernels)
    - (CUDA, HIP) [generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
    - (CUDA, HIP) [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
    - (CUDA) [xformers](https://github.com/facebookresearch/xformers)
    - (CUDA) [flash-attention](https://github.com/Dao-AILab/flash-attention)
    - (CUDA) [FBGEMM](https://github.com/pytorch/FBGEMM)
    - (CUDA) [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
    - (CUDA) [cutlass-kernels](https://github.com/ColfaxResearch/cutlass-kernels)
    
- Folder structure
    - operators: different operators to choose from
  ```bash    
        ├── addmm
        ├── bf16xint16_gemm
        ├── cross_entropy
        ├── decoding_attention
        ├── embedding
        ├── flash_attention
        ├── fp8_attention
        ├── fp8_fused_quant_gemm_rowwise
        ├── fp8_gemm
        ├── fp8_gemm_blockwise
        ├── fp8_gemm_rowwise
        ├── fused_linear_cross_entropy
        ├── fused_linear_jsd
        ├── gather_gemv
        ├── geglu
        ├── gemm
        ├── grouped_gemm
        ├── int4_gemm
        ├── jagged_layer_norm
        ├── jagged_mean
        ├── jagged_softmax
        ├── jagged_sum
        ├── jsd
        ├── kl_div
        ├── launch_latency
        ├── layer_norm
        ├── low_mem_dropout
        ├── mixed_gemm
        ├── op.py
        ├── op_task.py
        ├── ragged_attention
        ├── rms_norm
        ├── rope
        ├── softmax
        ├── sum
        ├── swiglu
        ├── template_attention
        ├── test_op
        ├── vector_add
        └── welford
  ```
    - kernels: Triton implementation of the Flash Attention v2
    - benchmarks: Perform simple benchmarks indicating tflops
  ```bash
        ├── compile_time
        ├── flash_attention_bench
        ├── gemm_bench
        └── nightly
  ```  
    - tests:

- Arguments `run.py`
    - `-op` - Operators to benchmark. Split with commas if multiple.
    - `-op-collection` - Operator collections to benchmark. Conflicts with `-op`. Choices: `default`, `liger`, `all`.
    - `-mode` - Test mode. Choices: `fwd`, `bwd`, `fwd_bwd`, `fwd_no_grad`. Default: `fwd`.
    - `-bwd` - Run backward pass.
    - `-fwd-bwd` - Run both forward and backward passes.
    - `-fwd-no-grad` - Run forward pass without gradients.
    - `-precision`, `-dtype` - Operator input precision. Default: `bypass`. Choices: `{AVAILABLE_PRECISIONS}`.
      
        ```bash
        # tritonbench/tritonbench/utils/env_utils.py
        AVAILABLE_PRECISIONS = [
            "bypass",    
            "fp32",    
            "tf32",    
            "fp16",    
            "amp",    
            "fx_int8",
            "bf16",    
            "amp_fp16",
            "amp_bf16",
            "fp8",
            ]
        ```
        
    - `-device` - Device to benchmark. Default: `cuda`.
    - `-warmup` - Number of warmup runs per benchmark. Default: `DEFAULT_WARMUP=25`.
    - `-iter` - Number of iterations per benchmark. Default: `DEFAULT_RUN_ITERS= 100`.
    - `-csv` - Print result as CSV.
    - `-output-dir` - Output result CSV to the specified directory.
    - `-output` - Output result CSV to a file.
    - `-output-json` - Output result JSON to a file.
    - `-skip-print` - Skip printing results.
    - `-plot` - Plot the results.
    - `-ci` - Run in CI mode.
    - `-metrics` - Metrics to collect, separated by commas. Example: `latency,tflops,speedup`.
    - `-metrics-gpu-backend` - Backend for GPU metrics collection. Choices: `torch`, `nvml`. Default: `torch`.
    - `-only` - Specify kernel implementations to run.
    - `-skip` - Specify kernel implementations to skip.
    - `-baseline` - Override the default baseline.
    - `-num-inputs` - Number of example inputs.
    - `-keep-going` - Continue execution even if errors occur.
    - `-input-id` - Start input ID to run.
    - `-test-only` - Run in test mode, skipping expensive steps like autotuning.
    - `-dump-ir` - Dump Triton IR.
    - `-gpu-lockdown` - Lock GPU frequency and clocks.
    - `-operator-loader` - Benchmark ATen operators in `tritonbench/operator_loader`.
    - `-cudagraph` - Benchmark with CUDA Graph.
    - `-isolate` - Run each operator in a separate process.
    - `-bypass-fail` - Continue execution even if an operator fails.
    - `-shuffle-shapes` - Randomly shuffle inputs before benchmarking.
    - `-compile-cold-start` - Include cold start time in compilation metrics.
- examples
  
    ```bash
    python run.py --m 4096 \
    --n 4096 \
    --k 4096 \
    --precision fp16 \
    --only triton_tutorial_matmul \
    --metrics tflops
    ```
    
    ```bash
    python run.py --op fp8_gemm --mode fwd --device cuda --metrics tflops
    
    ```
    

## [Simple-gpt](https://github.com/antferdom/simple-gpt)

## [torchtitan](https://github.com/pytorch/torchtitan)

### local installation

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
# CARE: using cu126 instead 124
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
# Download the tokenizer
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"
# HF_HOME location
export HF_HOME=<HF_HOME_LOCATION>
# Example workload
CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh
```

### **Dockerfile**

Caveats:

- `HF_TOKEN` needs to be specified in the host
- `WORKDIR` specified for torch-titan commands
- Examples of commands as comments in the end
- `<local_HF_HOME>:<container_HF_HOME>` loading from local nvme

```docker
# Use NVIDIA's official CUDA 12.4 base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
#FROM ubuntu:22.04

ARG HF_TOKEN    
ENV HF_TOKEN=${HF_TOKEN}

# makes sure the shell used for subsequent RUN commands is exactly Bash, as located in /bin.
SHELL ["/bin/bash", "-c"]

# Install dependencies
# llamacpp gcc compilation tools
RUN apt-get update && apt-get install -y \
    build-essential \
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
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev
    # Cleanup command to remove the apt cache and reduce the image size: # IMPORTANT: Enforces using sudo apt update when entering the container
    #&& rm -rf /var/lib/apt/lists/*

# Cloning the repo
RUN git clone https://github.com/pytorch/torchtitan
RUN pip install -r torchtitan/requirements.txt
# CARE: using cu126 instead 124
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

# Change to the repo directory using WORKDIR
WORKDIR /workspace/torchtitan

# Download the tokenizer
RUN python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"

# docker build . --build-arg HF_TOKEN="$HF_TOKEN" -t torchtitan_cuda126_torch26
# docker run --gpus all --shm-size 32g --network=host <local_HF_HOME>:<container_HF_HOME> --name torchtitan_workload -it --rm --ipc=host torchtitan_cuda126_torch26 bash -c 'CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh'
```

## [Meta-lingua](https://github.com/facebookresearch/lingua)

## Sglang DeepSeekv3

## Baselines

GPU metrics:

[torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/metrics.md): GPU Memory, Token per second per device([tps](https://github.com/pytorch/torchtitan/blob/fb0a942f1ff25db7512c814b413226a53cd83c4a/train.py#L360)) and mfu

tritonbench: FLOPS, latency, and speedup (relative to other kernels)

CPU metrics: CPU usage, memory used

### torchtitan

- Workload: train_configs/llama3_8b.toml with 100 steps
  
    ```bash
    # torchtitan Config.toml
    # NOTE: this toml config is a preset for 64 A100 GPUs.
    
    [job]
    dump_folder = "./outputs"
    description = "Llama 3 8B training"
    
    [profiling]
    enable_profiling = true
    save_traces_folder = "profile_trace"
    profile_freq = 100
    
    [metrics]
    log_freq = 10
    enable_tensorboard = true
    save_tb_folder = "tb"
    
    [model]
    name = "llama3"
    flavor = "8B"
    norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm
    tokenizer_path = "./torchtitan/datasets/tokenizer/original/tokenizer.model"
    
    [optimizer]
    name = "AdamW"
    lr = 3e-4
    
    [training]
    batch_size = 1
    seq_len = 8192
    warmup_steps = 200  # lr scheduler warm up
    max_norm = 1.0  # grad norm clipping
    steps = 100
    data_parallel_replicate_degree = 1
    data_parallel_shard_degree = -1
    tensor_parallel_degree = 1
    compile = false
    dataset = "c4"
    
    [experimental]
    context_parallel_degree = 1
    pipeline_parallel_degree = 1
    
    [checkpoint]
    enable_checkpoint = false
    folder = "checkpoint"
    interval_type = "steps"
    interval = 500
    model_weights_only = false
    export_dtype = "float32"
    async_mode = "disabled" # ["disabled", "async", "async_with_pinned_mem"]
    
    [activation_checkpoint]
    mode = 'selective'  # ['none', 'selective', 'full']
    selective_ac_option = 'op'  # 'int' = ac every positive int layer or 'op', ac based on ops policy
    
    [float8]
    enable_float8_linear = false
    ```
    
- Results
  
    ```bash
    nohup: ignoring input
    + NGPU=8
    + LOG_RANK=0
    + CONFIG_FILE=./train_configs/llama3_8b.toml
    + overrides=
    + '[' 0 -ne 0 ']'
    + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    + torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama3_8b.toml
    W0213 11:33:04.996000 14767 site-packages/torch/distributed/run.py:792] 
    W0213 11:33:04.996000 14767 site-packages/torch/distributed/run.py:792] *****************************************
    W0213 11:33:04.996000 14767 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
    W0213 11:33:04.996000 14767 site-packages/torch/distributed/run.py:792] *****************************************
    [rank0]:2025-02-13 11:33:11,198 - root - INFO - Starting job: Llama 3 8B training
    [rank0]:2025-02-13 11:33:12,134 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
    [rank0]:2025-02-13 11:33:12,138 - root - INFO - CUDA capacity: NVIDIA H200 with 139.83GiB memory
    [rank0]:2025-02-13 11:33:12,140 - root - WARNING - Error running lspci: [Errno 2] No such file or directory: 'lspci', fallback to use device_name
    [rank0]:2025-02-13 11:33:12,140 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
    [rank0]:2025-02-13 11:33:12,140 - root - INFO - Building 1-D device mesh with ['dp_shard'], [8]
    [rank0]:2025-02-13 11:33:18,527 - root - INFO - Building tiktoken tokenizer locally from ./torchtitan/datasets/tokenizer/original/tokenizer.model
    [rank0]:2025-02-13 11:33:18,746 - root - INFO - TikTokenizer built: #words 128256, BOS ID 128000, EOS ID 128001
    [rank0]:2025-02-13 11:33:18,746 - root - INFO - Preparing c4 dataset from allenai/c4
    [rank0]:2025-02-13 11:33:28,973 - root - INFO - Building llama3 8B with ModelArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256, multiple_of=1024, ffn_dim_multiplier=1.3, norm_eps=1e-05, rope_theta=500000, max_seq_len=8192, depth_init=True, norm_type='rmsnorm')
    [rank0]:2025-02-13 11:33:29,156 - root - INFO - Model llama3 8B [31msize: 8,030,261,248 total parameters
    [rank0]:2025-02-13 11:33:29,157 - root - INFO - Applied selective activation checkpointing to the model
    [rank0]:2025-02-13 11:33:29,239 - root - INFO - Applied FSDP to the model
    [rank0]:2025-02-13 11:33:29,478 - root - INFO - CUDA memory usage for model: 3.77GiB(2.70%)
    [rank0]:2025-02-13 11:33:29,482 - root - INFO - TensorBoard logging enabled. Logs will be saved at ./outputs/tb/20250213-1133
    [rank0]:2025-02-13 11:33:29,482 - root - INFO - Training starts at step 1, with local batch size 1, global batch size 8, sequence length 8192, total steps 100 (warmup 200)
    [rank0]:2025-02-13 11:33:29,482 - root - INFO - Profiling active. Traces will be saved at ./outputs/profile_trace
    [rank0]:2025-02-13 11:33:35,026 - root - INFO - step:  1  loss: 12.2417  memory: 42.08GiB(30.10%)  tps: 1,478  mfu: 8.65%
    [rank0]:2025-02-13 11:33:35,027 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
    [rank0]:2025-02-13 11:33:45,521 - root - INFO - step: 10  loss:  9.8821  mmemory: 49.59GiB(35.46%)  tps: 7,026  mfu: 41.15%
    [rank0]:2025-02-13 11:33:57,093 - root - INFO - step: 20  loss:  8.3854  memory: 49.59GiB(35.46%)  tps: 7,080  mfu: 41.46%
    [rank0]:2025-02-13 11:34:08,663 - root - INFO - step: 30  loss:  7.6991  memory: 49.59GiB(35.46%)  tps: 7,082  mfu: 41.47%
    [rank0]:2025-02-13 11:34:20,249 - root - INFO - step: 40  loss:  7.3925  memory: 49.59GiB(35.46%)  tps: 7,072  mfu: 41.41%
    [rank0]:2025-02-13 11:34:31,859 - root - INFO - step: 50  loss:  7.0563  memory: 49.59GiB(35.46%)  tps: 7,057  mfu: 41.32%
    [rank0]:2025-02-13 11:34:43,479 - root - INFO - step: 60  loss:  6.8581  memory: 49.59GiB(35.46%)  tps: 7,051  mfu: 41.29%
    [rank0]:2025-02-13 11:34:55,115 - root - INFO - step: 70  loss:  6.9885  memory: 49.59GiB(35.46%)  tps: 7,042  mfu: 41.24%
    [rank0]:2025-02-13 11:35:06,760 - root - INFO - step: 80  loss:  6.6429  memory: 49.59GiB(35.46%)  tps: 7,036  mfu: 41.20%
    [rank0]:2025-02-13 11:35:18,410 - root - INFO - step: 90  loss:  6.7054  memory: 49.59GiB(35.46%)  tps: 7,033  mfu: 41.19%
    [rank0]:2025-02-13 11:35:30,360 - root - INFO - step: 100  loss:  6.4719  memory: 49.59GiB(35.46%)  tps: 6,856  mfu: 40.15%
    [rank0]:2025-02-13 11:35:30,986 - root - INFO - Dumping profiler traces at step 100
    [rank0]:2025-02-13 11:35:31,218 - root - INFO - Finished dumping profiler traces in 0.23 seconds
    [rank0]:2025-02-13 11:35:31,219 - root - INFO - Sleeping 2 seconds for other ranks to complete
    [rank0]:2025-02-13 11:35:33,220 - root - INFO - Training completed
    ```
    

### tritonbench

- Workloads: [gemm](https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/gemm), [decoding_attention](https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/decoding_attention), [flash_attention](https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/flash_attention), [layer_norm](https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/layer_norm), [rms_norm](https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/rms_norm), [softmax](https://github.com/pytorch-labs/tritonbench/tree/main/tritonbench/operators/softmax).
  
    Any specific matrix dimension?
    
    gemm:
    
    ```bash
    python run.py --op gemm --mode fwd --device cuda --metrics latency,tflops,speedup --output-json "baseline_gemm.jsonl"
    ```
    
    decoding_attention:
    
    ```bash
    python run.py --op decoding_attention --mode fwd --device cuda --metrics latency,tflops,speedup --output-json "baseline_decoding_attention.jsonl"
    ```
    
    - ERROR: NameError: name 'fmha' is not defined
      
        ```bash
        Traceback (most recent call last):
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 125, in <module>
            run()
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 121, in run
            _run(args, extra_args)
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 35, in _run
            Opbench = load_opbench_by_name(args.op)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/op.py", line 70, in load_opbench_by_name
            module = importlib.import_module(f"..{op_pkg}", package=__name__)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/miniconda3/envs/tritonbench/lib/python3.11/importlib/__init__.py", line 126, in import_module
            return _bootstrap._gcd_import(name[level:], package, level)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
          File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
          File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
          File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
          File "<frozen importlib._bootstrap_external>", line 940, in exec_module
          File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/decoding_attention/__init__.py", line 1, in <module>
            from .operator import Operator
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/decoding_attention/operator.py", line 132, in <module>
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
            ^^^^
        NameError: name 'fmha' is not defined
        (tritonbench) rodri@cluster-h200-01-f2:~/tritonbench$ python run.py --op decoding_attentio --device cuda --metrics tflops speedup --output-json "baseli
        ne_decoding_attention.jsonl"
        Traceback (most recent call last):
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 125, in <module>
            run()
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 121, in run
            _run(args, extra_args)
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 35, in _run
            Opbench = load_opbench_by_name(args.op)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/op.py", line 60, in load_opbench_by_name
            raise RuntimeError(f"{op_name} is not found in the Tritonbench operator list.")
        RuntimeError: decoding_attentio is not found in the Tritonbench operator list.
        (tritonbench) rodri@cluster-h200-01-f2:~/tritonbench$ python run.py --op decoding_attention --device cuda --metrics tflops speedup --output-json "basel
        ine_decoding_attention.jsonl"
        Traceback (most recent call last):
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 125, in <module>
            run()
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 121, in run
            _run(args, extra_args)
          File "/mnt/co-research/home/rodri/tritonbench/run.py", line 35, in _run
            Opbench = load_opbench_by_name(args.op)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/op.py", line 70, in load_opbench_by_name
            module = importlib.import_module(f"..{op_pkg}", package=__name__)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/miniconda3/envs/tritonbench/lib/python3.11/importlib/__init__.py", line 126, in import_module
            return _bootstrap._gcd_import(name[level:], package, level)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
          File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
          File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
          File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
          File "<frozen importlib._bootstrap_external>", line 940, in exec_module
          File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/decoding_attention/__init__.py", line 1, in <module>
            from .operator import Operator
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/decoding_attention/operator.py", line 132, in <module>
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
            ^^^^
        NameError: name 'fmha' is not defined
        ```
        
    
    flash_attn:
    
    ```bash
    python run.py --op flash_attention --mode fwd --device cuda --metrics latency,tflops,speedup --output-json "baseline_flash.jsonl"
    ```
    
    - ERROR: NameError: name 'make_packed_qkv' is not defined
      
        ```bash
        TMA benchmarks will be running with experimental grid constant TMA descriptor.
          0%|                                                                                                                            | 0/8 [00:00<?, ?it/s]
        Caught exception, terminating early with partial results
        Traceback (most recent call last):
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 833, in run
            y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(
                                                          ^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 821, in _reduce_benchmarks
            acc[bm_name] = self._do_bench(
                           ^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 1076, in _do_bench
            fn = self._get_bm_func(fn_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 716, in _get_bm_func
            fwd_fn = fwd_fn_lambda(*self.example_inputs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 532, in _inner
            return function(self, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/flash_attention/operator.py", line 254, in flash_v2
            qkv = make_packed_qkv(q, k, v)
                  ^^^^^^^^^^^^^^^
        NameError: name 'make_packed_qkv' is not defined
          (Batch, Heads, SeqLen, Dhead)
        -------------------------------
        ```
        
    
    layer_norm:
    
    ```bash
    python run.py --op layer_norm --mode fwd --device cuda --metrics latency,tflops,speedup --output-json "baseline_layernorm.jsonl"
    ```
    
    rms_norm:
    
    ```bash
    python run.py --op rms_norm --mode fwd --device cuda --metrics latency,tflops,speedup --output-json "baseline_rmsnorm.jsonl"
    ```
    
    - ERROR: 'NoneType' object is not callable
      
        ```bash
        Caught exception, terminating early with partial results
        Traceback (most recent call last):
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 833, in run
            y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(
                                                          ^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 821, in _reduce_benchmarks
            acc[bm_name] = self._do_bench(
                           ^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 1076, in _do_bench
            fn = self._get_bm_func(fn_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 716, in _get_bm_func
            fwd_fn = fwd_fn_lambda(*self.example_inputs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/utils/triton_op.py", line 532, in _inner
            return function(self, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/mnt/co-research/home/rodri/tritonbench/tritonbench/operators/rms_norm/operator.py", line 62, in liger_rms
            self.liger_rms_op = LigerRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        TypeError: 'NoneType' object is not callable
          (M, H)
        --------
        ```
        
    
    softmax:
    
    ```bash
    python run.py --op softmax --mode fwd --device cuda --metrics latency,tflops,speedup --output-json "baseline_softmax.jsonl"
    ```
