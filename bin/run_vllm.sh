python -m vllm.entrypoints.openai.api_server \
       --model ./gemma2-9b-instruct-dpo-awq  \
       --max_model_len 8192 \
       --quantization awq \
       --host 0.0.0.0 \
       --port 8000 \
       --trust-remote-code