#!/usr/bin/env python
# coding: utf-8

"""
vLLM Launch Server
=================
This script launches vLLM as a local Python server and
provides an HTTP endpoint '/generate' through FastAPI (uvicorn).

Usage example:
    CUDA_VISIBLE_DEVICES=0 python3 minimal_r1/launch_vllm.py --model_name Seungyoun/Qwen2.5-7B-Open-R1-Distill

After server launch, access http://localhost:8000/docs for Swagger UI testing.
"""

import uvicorn
import argparse
import io
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List

# vLLM
from vllm import LLM, SamplingParams

# ----------- Argument Parsing -----------
parser = argparse.ArgumentParser(description="Launch vLLM Server with a specified model.")
parser.add_argument("--model_name", type=str, required=True, help="Path or name of the model to use")
args = parser.parse_args()


# ----------- FastAPI init -----------
app = FastAPI(title="vLLM Server", version="0.1")


# ----------- vLLM init -----------
print(f"args.model_name: {args.model_name}")
llm = LLM(
    model=args.model_name,
    trust_remote_code=True,
    tensor_parallel_size=2,  # add this if you want to use multiple GPUs
    dtype="bfloat16"
)


class GenerateRequest(BaseModel):
    prompts: List[str]
    num_gen: int = 1
    temperature: float = 0.8
    max_tokens: int = 128


class GenerateResponse(BaseModel):
    generations: List[List[str]]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """vLLM으로 num_gen개씩 샘플 생성"""
    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
        max_tokens=req.max_tokens,
        n = req.num_gen,
        # stop=["</answer>"]
    )

    num_prompts = len(req.prompts)  # Original number of prompts
    outputs = llm.generate(req.prompts, sampling_params)  # Ensured multiplication

    # Initialize the correct number of lists
    generations = [[] for _ in range(num_prompts)]

    # Correctly distribute outputs back into the corresponding prompt group
    for i, output in enumerate(outputs):
        prompt_idx = i % num_prompts  # Ensures proper grouping
        for _output in output.outputs:
            generations[prompt_idx].append(_output.text)  # Append to corresponding list

    return GenerateResponse(generations=generations)



@app.post("/load_weights")
async def load_weights(request: Request):
    """
    Load PyTorch state_dict to vLLM model.
    Client can send state_dict in the following way:

    ```python
    import requests, torch, io

    # state_dict (예: fine-tuned weights)
    model_sd = torch.load("fine_tuned.pt")

    buf = io.BytesIO()
    torch.save(model_sd, buf)
    buf.seek(0)
    resp = requests.post("http://localhost:8000/load_weights", data=buf.read())

    print(resp.json())
    ```
    """
    if llm is None:
        raise HTTPException(status_code=400, detail="LLM is not initialized.")

    try:
        weights_data = await request.body()
        buffer = io.BytesIO(weights_data)

        state_dict = torch.load(buffer, map_location="cpu")  # change to state_dict

        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
        print("\3[32mNew model weights loaded.\3[0m")

        return {"status": "success", "message": "Model weights loaded."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load weights: {str(e)}")


if __name__ == "__main__":
    print("Launching vLLM API server on :8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
