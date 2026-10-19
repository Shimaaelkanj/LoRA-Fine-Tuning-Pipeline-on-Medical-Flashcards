
#!/bin/bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules medical=deploy/adapters/medical-rank32 \
  --port 8000