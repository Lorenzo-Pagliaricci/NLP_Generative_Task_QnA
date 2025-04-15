mlx_lm.generate \
    --model ./proveIgnazio_MLX_LM/models/base/Llama-3.2-1B-Instruct-4bit \
	--adapter-path ./proveIgnazio_MLX_LM/models/adapters/adapter_Llama-3.2-1B-Instruct-4bit_2bs_4ls \
    --prompt "Is there any relationship between histone ubiquitylation and splicing?" \
    --max-tokens 1000