# Reference commands to run benchmark

## Setup the openai api
```shell
tmux new-session -d -s fastchat-controller 'python -m fastchat_serve.controller --host 0.0.0.0'
tmux new-session -d -s fastchat-vi-7b 'python -m fastchat_serve.multi_model_worker --config="/app/third_party/fastchat-serve/src/fastchat_serve/config/multi_model_worker.json"'
tmux new-session -d -s fastchat-openai 'python -m fastchat_serve.openai.api_server --host 0.0.0.0 --port 6888'
```

Currently benchmark with [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).


```shell
python benchmark_openai_api.py
python benchmark_openai_api.py --model "Mistral-7B-Instruct-v0.1"
```
