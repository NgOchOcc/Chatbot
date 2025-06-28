python src/quantize.py \
    --model OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2024-10-09 \
    --output ./gemma2-9b-instruct-dpo-awq \
    --w_bit 4 \
    --q_group_size 128