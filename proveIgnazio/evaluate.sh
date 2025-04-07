mlx_lm.lora \
    --model ./proveIgnazio/models/base/Llama-3.2-1B-Instruct-4bit \
    --adapter-path ./proveIgnazio/models/adapters/adapter_Llama-3.2-1B-Instruct-4bit_2bs_4ls \
    --data proveIgnazio/data/processed \
    --test \
    --test-batches 2
    # --batch-size 1 \
    # --num-layers 4 \
    # --iters 500 \
    # --steps-per-eval 10