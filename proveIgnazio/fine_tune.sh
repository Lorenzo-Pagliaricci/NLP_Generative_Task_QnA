


mlx_lm.lora \
    --model mlx-community/gemma-3-text-12b-it-4bit \
    --train \
    --batch-size 20 \
    --num-layers 32 \
    --data data/processed \
    --iters 60