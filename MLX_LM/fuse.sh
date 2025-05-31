mlx_lm.fuse \
    --model ./MLX_LM/models/base/Phi-3-mini-128k-instruct-4bit \
    --adapter-path ./MLX_LM/models/adapters/adapter_Phi-3-mini-128k-instruct-4bit_2bs_4ls \
    --save-path ./MLX_LM/models/fused/fused_Phi-3-mini-128k-instruct-4bit_2bs_4ls 