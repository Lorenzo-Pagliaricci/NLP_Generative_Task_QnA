# --config proveIgnazio_MLX_LM/data/dataset.yaml


# huggingface-cli download mlx-community/gemma-3-1b-it-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/gemma-3-1b-it-4bit
# huggingface-cli download mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/DeepSeek-R1-Distill-Qwen-1.5B-4bit
# huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-1M-3bit  --local-dir ./proveIgnazio_MLX_LM/models/base/Qwen2.5-7B-Instruct-1M-3bit
# huggingface-cli download mlx-community/Ministral-8B-Instruct-2410-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/Ministral-8B-Instruct-2410-4bit
# huggingface-cli download mlx-community/Llama-4-Maverick-17B-16E-Instruct-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/Llama-4-Maverick-17B-16E-Instruct-4bit
# 2_BEST # huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/Llama-3.2-1B-Instruct-4bit
# huggingface-cli download mlx-community/Qwen2.5-1.5B-Instruct-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/Qwen2.5-1.5B-Instruct-4bit
# 1_BEST # huggingface-cli download mlx-community/Phi-3-mini-4k-instruct-4bit  --local-dir ./proveIgnazio_MLX_LM/models/base/Phi-3-mini-4k-instruct-4bit



mlx_lm.lora \
    --model ./proveIgnazio_MLX_LM/models/base/Phi-3-mini-4k-instruct-4bit \
    --train \
	--adapter-path ./proveIgnazio_MLX_LM/models/adapters/adapter_Phi-3-mini-4k-instruct-4bit_2bs_4ls \
    --batch-size 2 \
    --val-batches 2 \
    --num-layers 4 \
    --data proveIgnazio_MLX_LM/data/processed \
    --iters 500 \
    --steps-per-eval 10 \
    --grad-checkpoint