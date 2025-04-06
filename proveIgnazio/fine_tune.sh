# --config proveIgnazio/data/dataset.yaml


# huggingface-cli download mlx-community/gemma-3-1b-it-4bit  --local-dir ./models/base/gemma-3-1b-it-4bit

mlx_lm.lora \
    --model ./proveIgnazio/models/base/gemma-3-1b-it-4bit \
    --train \
	--adapter-path ./models/adapters/adapter_gemma-3-1b-it-4bit \
    --batch-size 1 \
    --num-layers 4 \
    --data proveIgnazio/data/processed \
    --iters 500 \
    --step-per-eval 10