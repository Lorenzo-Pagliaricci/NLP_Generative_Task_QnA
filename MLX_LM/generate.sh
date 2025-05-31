mlx_lm.generate \
    --model ./MLX_LM/models/base/Phi-3-mini-128k-instruct-4bit \
	--adapter-path ./MLX_LM/models/adapters/adapter_Phi-3-mini-128k-instruct-4bit_3bs_6ls \
    --prompt "Is there any relationship between histone ubiquitylation and splicing?" \
    --max-tokens 1000