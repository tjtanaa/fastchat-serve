import os
import openai
import argparse
import time
import requests
from urllib.parse import urlparse
import json
import random
import pickle
from typing import List, Tuple, Dict
from loguru import logger

def cli_args():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--api-base", type=str, default="http://localhost:6888/v1")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--model", type=str, default="vicuna-7b-v1.5")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-prompts",
                        type=int,
                        default=100,
                        help="Number of prompts to process.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    args = parser.parse_args()
    return args


def get_token_length(
    prompts:List[str],
    model: str,
    token_check_url: str,
    headers: Dict
) -> List[int]:

    data_json = {
        "prompts": [
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 0
            }
            for prompt in prompts
        ]
    }
    response = requests.post(
        token_check_url,
        headers=headers, 
        data=json.dumps(data_json)

    )

    response_json = response.json()

    token_length = [token_info["tokenCount"] for token_info in response_json["prompts"]]

    return token_length

def sample_requests(
    dataset_path: str,
    num_requests: int,
    token_check_url: str,
    headers: Dict,
    model: str

) -> List[Tuple[str, int, int]]:
    
    cached_dataset_path = dataset_path + "_" + model + ".pkl"

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []

    if os.path.exists(cached_dataset_path):
        logger.info("Loading Cached Dataset")
        with open(cached_dataset_path, "rb") as f:
            filtered_dataset = pickle.load(f)
        logger.info("Loaded Cached Dataset")
    else:
        logger.info("Loading Dataset")
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data["conversations"][0]["value"],
                    data["conversations"][1]["value"]) for data in dataset]

        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _ in dataset]
        # prompt_token_ids = tokenizer(prompts).input_ids
        prompts_length = get_token_length(prompts, model, token_check_url, headers)
        completions = [completion for _, completion in dataset]
        # completion_token_ids = tokenizer(completions).input_ids
        completions_length = get_token_length(completions, model, token_check_url, headers)
        tokenized_dataset = []
        for i in range(len(dataset)):
            prompt_len = prompts_length[i]
            output_len = completions_length[i]

            tokenized_dataset.append((prompts[i], prompt_len, output_len))

        for prompt, prompt_len, output_len in tokenized_dataset:
            if prompt_len < 4 or output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append((prompt, prompt_len, output_len))


            # Open a file and use dump() 
            with open(cached_dataset_path, 'wb') as file: 
                
                # A new file will be created 
                pickle.dump(filtered_dataset, file) 

        logger.info("Loaded Dataset")
    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)
    sampled_requests = filtered_dataset[:num_requests]
    return sampled_requests


def main(args):

    openai.api_key = "EMPTY"    
    openai.api_base = args.api_base
    
    parsed = urlparse(args.api_base)
    print(parsed)
    token_check_url = parsed.scheme + "://" + parsed.netloc + '/api/v1/token_check'
    headers = {"Content-Type": "application/json"}

    requests = sample_requests(args.dataset, args.num_prompts, token_check_url, headers, model=args.model)



    elapsed_time = 0.0
    response_list = []

    if args.mode == "chat":

        for i in range(len(requests)):
            prompt, prompt_len, output_len = requests[i]
            t = time.time()
            full_response = ""
            
            for response in openai.ChatCompletion.create(
                model=args.model,
                messages=(
                [
                    {"role": "user", "content": prompt}
                ]
                ),
                stream=True,
                temperature=args.temperature,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                max_tokens=args.max_tokens,
            ):
                full_response += response.choices[0].delta.get("content", "")

            duration = time.time() - t
            elapsed_time += duration
            response_list.append(full_response)

    else:
        raise NotImplementedError("Only support benchmarking in chatmode")

    output_token_length = get_token_length(
        full_response,
        args.model,
        token_check_url,
        headers
    )

    total_num_tokens = sum(prompt_len + output_token_length[i]
                           for i, (_, prompt_len, _) in enumerate(requests))
    logger.info(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":

    args = cli_args()
    main(args)