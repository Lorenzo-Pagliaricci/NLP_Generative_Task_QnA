mlx_lm.lora \
    --model ./MLX_LM/models/base/Phi-3-mini-4k-instruct-4bit \
    --adapter-path ./MLX_LM/models/adapters/adapter_Phi-3-mini-4k-instruct-4bit_2bs_4ls \
    --data MLX_LM/data/processed \
    --test \
    --test-batches 1
    # --batch-size 1 \
    # --num-layers 4 \
    # --iters 500 \
    # --steps-per-eval 10