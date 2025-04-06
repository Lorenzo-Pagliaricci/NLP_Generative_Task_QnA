mlx_lm.lora \
    --model ./proveIgnazio/models/base/gemma-3-1b-it-4bit \
    --adapter-path ./proveIgnazio/models/adapters/adapter_gemma-3-1b-it-4bit \
    --data proveIgnazio/data/processed \
    --test 
    # --batch-size 1 \
    # --num-layers 4 \
    # --iters 500 \
    # --steps-per-eval 10