#!/bin/bash
python3 -m pip install pydantic==2.4.2
python3 -m pip install -e .
python3 -m fastchat_serve.multi_model_worker --config="config/multi_model_worker.json"