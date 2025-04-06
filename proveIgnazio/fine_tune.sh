mlx_lm.lora \
    --model mlx-community/gemma-3-text-12b-it-4bit \
    --train \
    --batch-size 1 \
    --num-layers 4 \
    --data data/processed/bioasq_qa.jsonl \
    --iters 10 