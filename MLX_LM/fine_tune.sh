# --config MLX_LM/data/dataset.yaml


# huggingface-cli download mlx-community/gemma-3-1b-it-4bit  --local-dir ./MLX_LM/models/base/gemma-3-1b-it-4bit
# huggingface-cli download mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit  --local-dir ./MLX_LM/models/base/DeepSeek-R1-Distill-Qwen-1.5B-4bit
# huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-1M-3bit  --local-dir ./MLX_LM/models/base/Qwen2.5-7B-Instruct-1M-3bit
# huggingface-cli download mlx-community/Ministral-8B-Instruct-2410-4bit  --local-dir ./MLX_LM/models/base/Ministral-8B-Instruct-2410-4bit
# huggingface-cli download mlx-community/Llama-4-Maverick-17B-16E-Instruct-4bit  --local-dir ./MLX_LM/models/base/Llama-4-Maverick-17B-16E-Instruct-4bit
# 2_BEST # huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit  --local-dir ./MLX_LM/models/base/Llama-3.2-1B-Instruct-4bit
# huggingface-cli download mlx-community/Qwen2.5-1.5B-Instruct-4bit  --local-dir ./MLX_LM/models/base/Qwen2.5-1.5B-Instruct-4bit
# 1_BEST # huggingface-cli download mlx-community/Phi-3-mini-4k-instruct-4bit  --local-dir ./MLX_LM/models/base/Phi-3-mini-4k-instruct-4bit
# huggingface-cli download mlx-community/Phi-3-mini-128k-instruct-4bit --local-dir ./MLX_LM/models/base/Phi-3-mini-128k-instruct-4bit






mlx_lm.lora \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train \
	--adapter-path ./MLX_LM/models/adapters/adapter_Qwen2.5-0.5B-Instruct_2bs_6ls \
    --batch-size 3 \
    --val-batches 2 \
    --learning-rate 0.0001 \
    --num-layers 6 \
    --data MLX_LM/data/processed \
    --iters 200 \
    --steps-per-eval 50 \
    --seed 42 \
    --fine-tune-type lora \
    --grad-checkpoint \
    # --optimizer adam