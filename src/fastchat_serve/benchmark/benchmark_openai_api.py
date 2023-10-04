import openai
import argparse
import time
import requests
from urllib.parse import urlparse
import json
from loguru import logger

def cli_args():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--api-base", type=str, default="http://localhost:6888/v1")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--model", type=str, default="vicuna-7b-v1.5")
    parser.add_argument(
        "--prompt", type=str, default="Write a snake game."
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()
    return args

def main(args):

    openai.api_key = "EMPTY"    
    openai.api_base = args.api_base


    parsed = urlparse(args.api_base)
    print(parsed)

    token_check_url = parsed.scheme + "://" + parsed.netloc + '/api/v1/token_check'
    headers = {"Content-Type": "application/json"}

    if args.mode == "chat":
        full_response = ""
        
        t = time.time()
        for response in openai.ChatCompletion.create(
            model=args.model,
            messages=(
            [
                {"role": "user", "content": args.prompt}
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

    else:
        raise NotImplementedError("Only support benchmarking in chatmode")

    data_json = {
        "prompts": [
            {
                "model": args.model,
                "prompt": full_response,
                "max_tokens": 0
            }
        ]
    }
    response = requests.post(
        token_check_url,
        headers=headers, 
        data=json.dumps(data_json)

    )

    response_json  = response.json()
    speed = round(response_json["prompts"][0]["tokenCount"] / duration, 2)
    logger.info(f"""
        output: {full_response}


        speed (token/s): {speed}
    """)


if __name__ == "__main__":

    args = cli_args()
    main(args)